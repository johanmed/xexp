"""
Module with Set-to-Set Transformer for Gene Expression Modeling
Predict gene expression in target tissues from observed tissues using cross-attention
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SetTransformer(nn.Module):
    def __init__(
        self,
        n_tissues: int,
        n_genes: int,
        dims: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.n_tissues = n_tissues
        self.n_genes = n_genes
        self.dims = dims

        self.tissue_embedder = nn.Embedding(n_tissues, dims)
        self.gene_embedder = nn.Embedding(n_genes, dims)

        self.expression_embedder = nn.Sequential(
            nn.Linear(1, dims // 2),
            nn.LayerNorm(dims // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims // 2, dims),
        )

        # Encoder: Observation processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dims,
            nhead=n_heads,
            dim_feedforward=dims * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers, norm=nn.LayerNorm(dims)
        )

        # Decoder: Cross-attention from query to observations
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dims,
            nhead=n_heads,
            dim_feedforward=dims * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_decoder_layers, norm=nn.LayerNorm(dims)
        )

        # Output heads
        # Expression head
        self.expression_head = nn.Sequential(
            nn.Linear(dims, dims),
            nn.LayerNorm(dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims, dims // 2),
            nn.GELU(),
            nn.Linear(dims // 2, 1),
        )
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dims, dims // 2),
            nn.GELU(),
            nn.Linear(dims // 2, 1),
            nn.Softplus(),  # Ensure positive
        )

        self._init_weights()

    def _init_weights(self):
        # Embedding initialization
        nn.init.xavier_uniform_(self.gene_embedder.weight)
        nn.init.xavier_uniform_(self.tissue_embedder.weight)

        # Expression projection initialization
        for m in self.expression_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialization of other linear layers
        for p in self.parameters():
            if p.dim() > 1 and not isinstance(p, nn.Embedding):
                nn.init.xavier_uniform_(p)

    def encode_observations(
        self,
        obs_tissues: torch.Tensor,
        obs_genes: torch.Tensor,
        obs_expressions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode gene-tissue-expression (set) observations"""
        batch_size, n_obs = obs_tissues.shape

        tissue_embeds = self.tissue_embedder(obs_tissues)
        gene_embeds = self.gene_embedder(obs_genes)
        expression_embeds = self.expression_embedder(obs_expressions.unsqueeze(-1))

        combined = tissue_embeds + gene_embeds + expression_embeds
        encoded = self.encoder(combined, src_key_padding_mask=mask)

        return encoded

    def decode_query(
        self,
        query_tissues: torch.Tensor,
        query_genes: torch.Tensor,
        encoded_observations: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode queries using cross-attention over observations"""
        batch_size, n_queries = query_tissues.shape

        tissue_embeds = self.tissue_embedder(query_tissues)
        gene_embeds = self.gene_embedder(query_genes)

        query_features = tissue_embeds + gene_embeds
        decoded = self.decoder(
            tgt=query_features,
            memory=encoded_observations,
            memory_key_padding_mask=obs_mask,
        )

        return decoded

    def forward(
        self,
        obs_tissues: torch.Tensor,
        obs_genes: torch.Tensor,
        obs_expressions: torch.Tensor,
        query_tissues: torch.Tensor,
        query_genes: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            obs_*: Observed data tensors [batch, n_obs]
            query_*: Query tensors [batch, n_queries]
            obs_mask: Boolean mask, True indicates padding positions [batch, n_obs]
            return_attention: If True, return attention weights for explainability

        Returns:
            Dictionary with 'expression', 'uncertainty', and optionally 'attention_weights'
        """
        encoded = self.encode_observations(
            obs_tissues, obs_genes, obs_expressions, obs_mask
        )

        # Decode queries
        decoded = self.decode_query(query_tissues, query_genes, encoded, obs_mask)

        # Predict expression and uncertainty
        expressions = self.expression_head(decoded).squeeze(-1)
        uncertainties = self.uncertainty_head(decoded).squeeze(-1)

        outputs = {
            "expressions": expressions,
            "uncertainties": uncertainties,
            "encoded_observations": encoded,
            "decoded_query": decoded,
        }

        if return_attention:
            attention_weights = []

            def hook_fn(module, input, output):
                attention_weights.append(
                    output[1]
                )  # output is (attn_output, attn_weights)

            handles = []
            for layer in self.decoder.layers:
                handles.append(layer.multihead_attn.register_forward_hook(hook_fn))

            # Re-run decoder with hooks
            _ = self.decode_query(query_tissues, query_genes, encoded, obs_mask)

            for h in handles:
                h.remove()

            outputs["attention_weights"] = attention_weights

        return outputs
