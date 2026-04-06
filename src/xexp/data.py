"""Module with constructs for representing, processing and modeling gene expression data"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GeneExpression:
    tissue: int
    gene: int
    expression: float


@dataclass
class Example:
    observations: list[GeneExpression]
    query: list[GeneExpression]

    def to_tensor_dict(self) -> dict:
        """Convert to expected MicroarrayDataset format"""

        def to_tensors(obs_list):
            return {
                "tissues": torch.tensor([o.tissue for o in obs_list]),
                "genes": torch.tensor([o.gene for o in obs_list]),
                "expressions": torch.tensor(
                    [o.expression for o in obs_list], dtype=torch.float32
                ),
            }

        obs_tensors = to_tensors(self.observations)
        query_tensors = to_tensors(self.query)

        return {
            "obs_tissues": obs_tensors["tissues"],
            "obs_genes": obs_tensors["genes"],
            "obs_expressions": obs_tensors["expressions"],
            "query_tissues": query_tensors["tissues"],
            "query_genes": query_tensors["genes"],
            "targets": query_tensors["expressions"],
        }


class MicroarrayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        expression_matrix: np.ndarray,
        tissue_labels: np.ndarray,
        gene_labels: np.ndarray,
        mask_ratio=0.2,
    ):
        self.expressions = expression_matrix
        self.n_tissues = expression_matrix.shape[1]
        self.n_genes = expression_matrix.shape[2]
        self.tissues = tissue_labels
        self.genes = gene_labels
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Example:
        """
        Return an Example
        """
        sample = self.expressions[idx]

        n_observations = max(1, int(self.n_tissues * (1 - self.mask_ratio)))
        obs_tissue_ids = np.random.choice(self.n_tissues, n_observations, replace=False)
        query_tissue_ids = np.setdiff1d(np.arange(self.n_tissues), obs_tissue_ids)

        observed = []
        for t_idx in obs_tissue_ids:
            for g_idx in range(self.n_genes):
                observed.append(
                    GeneExpression(
                        gene=g_idx,
                        tissue=t_idx,
                        expression=sample[g_idx, t_idx],
                    )
                )

        # Predict all genes for one random query tissue
        target_tissue = np.random.choice(query_tissue_ids)
        query = []
        for g_idx in range(self.n_genes):
            query.append(
                GeneExpression(
                    gene=g_idx,
                    tissue=target_tissue,
                    expression=sample[g_idx, target_tissue],
                )
            )

        return Example(observations=observed, query=query)


def collate_fn(batch: List[Example]) -> dict:
    """
    Batch multiple Example into tensor dictionaries
    Handle padding for variable-length observations
    """
    tensor_dicts = [ex.to_tensor_dict() for ex in batch]

    max_obs_len = max(d["obs_genes"].shape[0] for d in tensor_dicts)
    max_query_len = max(d["query_genes"].shape[0] for d in tensor_dicts)

    batch_size = len(batch)

    # Initialize padded tensors
    batch_obs_tissues = torch.zeros(batch_size, max_obs_len, dtype=torch.long)
    batch_obs_genes = torch.zeros(batch_size, max_obs_len, dtype=torch.long)
    batch_obs_expressions = torch.zeros(batch_size, max_obs_len, dtype=torch.float32)
    batch_obs_mask = torch.ones(
        batch_size, max_obs_len, dtype=torch.bool
    )  # True = padding

    batch_query_tissues = torch.zeros(batch_size, max_query_len, dtype=torch.long)
    batch_query_genes = torch.zeros(batch_size, max_query_len, dtype=torch.long)
    batch_targets = torch.zeros(batch_size, max_query_len, dtype=torch.float32)

    # Fill in data
    for i, d in enumerate(tensor_dicts):
        obs_len = d["obs_genes"].shape[0]
        query_len = d["query_genes"].shape[0]

        batch_obs_tissues[i, :obs_len] = d["obs_tissues"]
        batch_obs_genes[i, :obs_len] = d["obs_genes"]
        batch_obs_expressions[i, :obs_len] = d["obs_expressions"]
        batch_obs_mask[i, :obs_len] = False  # Not padding

        batch_query_tissues[i, :query_len] = d["query_tissues"]
        batch_query_genes[i, :query_len] = d["query_genes"]
        batch_targets[i, :query_len] = d["targets"]

    return {
        "obs_tissues": batch_obs_tissues,
        "obs_genes": batch_obs_genes,
        "obs_expressions": batch_obs_expressions,
        "obs_mask": batch_obs_mask,
        "query_tissues": batch_query_tissues,
        "query_genes": batch_query_genes,
        "targets": batch_targets,
    }
