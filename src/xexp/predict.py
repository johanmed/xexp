"""Main script for prediction"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import MicroarrayDataset
from nn import SetTransformer


def predict_expression(
    model: SetTransformer,
    data: Dict[str, torch.Tensor],
    target_tissue: int = 0, # tissue 1
    gene_list: Optional[List[int]] = None,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Predict expression in a specific target tissue

    Args:
        model: Trained model
        data: Dict with 'obs_tissues', 'obs_genes', etc
        target_tissue: Tissue ID to predict
        gene_list: List of gene IDs to predict (default: all genes)

    Return:
        Dictionary with 'expression', 'uncertainty', 'lower_bound', 'upper_bound'
    """
    model = model.to(device)
    model.eval()

    if gene_list is None:
        gene_list = list(range(model.n_genes))

    obs_tissues = data["obs_tissues"].unsqueeze(0).to(device)
    obs_genes = data["obs_genes"].unsqueeze(0).to(device)
    obs_expressions = data["obs_expressions"].unsqueeze(0).to(device)

    query_tissues = (
        torch.LongTensor([target_tissue] * len(gene_list)).unsqueeze(0).to(device)
    )
    query_genes = torch.LongTensor(gene_list).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            obs_tissues=obs_tissues,
            obs_genes=obs_genes,
            obs_expressions=obs_expressions,
            query_tissues=query_tissues,
            query_genes=query_genes,
        )

    pred_expressions = outputs["expressions"].squeeze(0)
    pred_uncertainties = outputs["uncertainties"].squeeze(0)
    std = torch.sqrt(pred_uncertainties)

    return {
        "expressions": pred_expressions,
        "uncertainties": pred_uncertainties,
        "std": std,
        "lower_bound_95": pred_expressions - 1.96 * std,
        "upper_bound_95": pred_expressions + 1.96 * std,
    }


def explain_prediction(
    model: SetTransformer,
    data: Dict[str, torch.Tensor],
    tissue_labels: np.ndarray,
    gene_labels: np.ndarray,
    target_tissue_idx: int = 0, # tissue 1
    target_gene_idx: int = 0, # gene 1
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Return a dataframe showing which observed (gene, tissue) pairs 
    most influenced the prediction for a specific (target_gene, target_tissue)
    """

    model = model.to(device)
    model.eval()

    obs_tissues = data["obs_tissues"].unsqueeze(0).to(device)
    obs_genes = data["obs_genes"].unsqueeze(0).to(device)
    obs_expressions = data["obs_expressions"].unsqueeze(0).to(device)
    
    query_tissues = (
        torch.LongTensor([target_tissue_idx] * len(gene_labels)).unsqueeze(0).to(device)
    )
    query_genes = torch.LongTensor(gene_labels).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            obs_tissues=obs_tissues,
            obs_genes=obs_genes,
            obs_expressions=obs_expressions,
            query_tissues=query_tissues,
            query_genes=query_genes,
            return_attention=True
        )
    
    # Get attention from last decoder layer
    attn = outputs['attention_weights'][-1].squeeze(0)
    
    # Find the query index for our target (gene, tissue)
    query_mask = (
        (query_genes == target_gene_idx) &
        (query_tissues == target_tissue_idx)
    )

    matches = query_mask.nonzero(as_tuple=True)
    query_idx = matches[1][0].item()
    
    # Get attention weights for specific query
    query_attn = attn[query_idx].cpu().numpy()
    
    results = []
    for obs_idx in range(obs_genes.shape[1]):
        results.append({
            'obs_gene': gene_labels[obs_genes[0, obs_idx].item()],
            'obs_tissue': tissue_labels[obs_tissues[0, obs_idx].item()],
            'obs_expression': obs_expressions[0, obs_idx].item(),
            'attention_weight': query_attn[obs_idx],
            'pct_contribution': query_attn[obs_idx] / query_attn.sum() * 100
        })
        
    df = pd.DataFrame(results)
    return df.sort_values('attention_weight', ascending=False)
    

if __name__ == "__main__":
    RANDOM_SEED = 2026
    N_TISSUES = 10
    N_GENES = 1000

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_samples = 3
    expression_matrix = np.random.randn(n_samples, N_TISSUES, N_GENES)
    tissue_labels = np.random.randint(0, N_TISSUES, 10) # length label must match total number of tissues
    gene_labels = np.random.randint(0, N_GENES, 1000) # length label must match total number of genes
    
    test_dataset = MicroarrayDataset(
        expression_matrix=expression_matrix,
        tissue_labels=tissue_labels,
        gene_labels=gene_labels,
    )

    transformed_dataset = [d.to_tensor_dict() for d in test_dataset]

    model = SetTransformer(
        n_tissues=N_TISSUES,
        n_genes=N_GENES,
        dims=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
    )

    loaded_weights = torch.load("../../results/xexp_weights.pt", weights_only=True)
    model.load_state_dict(loaded_weights)

    print("Inference...")

    for data in transformed_dataset:
        predictions = predict_expression(
            model,
            data
        )
        print(f"Predicted expressions for tissue 1: {predictions['expressions']}")
        print(f"Predicted uncertainties for tissue 1: {predictions['uncertainties']}")
        explanations = explain_prediction(
            model,
            data,
            tissue_labels,
            gene_labels
        )
        print(f"Explanations for tissue 1 and gene 1\n: {explanations}")
