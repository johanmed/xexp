"""Main script for prediction"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MicroarrayDataset
from nn import SetTransformer


def predict_expression(
    model: SetTransformer,
    observations: Dict[str, torch.Tensor],
    target_tissue: int = 0,
    gene_list: Optional[List[int]] = None,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Inference function for predicting expression in a specific target tissue

    Args:
        model: Trained model
        observations: Dict with 'genes', 'tissues', 'expressions'
        target_tissue: Tissue ID to predict
        gene_list: List of gene IDs to predict (default: all genes)

    Returns:
        Dictionary with 'expression', 'uncertainty', 'lower_bound', 'upper_bound'
    """
    model = model.to(device)
    model.eval()

    if gene_list is None:
        gene_list = list(range(model.n_genes))

    obs_tissues = observations["obs_tissues"].unsqueeze(0).to(device)
    obs_genes = observations["obs_genes"].unsqueeze(0).to(device)
    obs_expressions = observations["obs_expressions"].unsqueeze(0).to(device)

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


if __name__ == "__main__":
    RANDOM_SEED = 2026
    N_TISSUES = 10
    N_GENES = 1000

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_samples = 3
    expression_matrix = np.random.randn(n_samples, N_TISSUES, N_GENES)
    tissue_labels = np.random.randint(0, N_TISSUES, n_samples)
    gene_labels = np.random.randint(0, N_GENES, n_samples)
    
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

        print(f"Predicted expressions: {predictions['expressions']}")
        print(f"Predicted uncertainties: {predictions['uncertainties']}")
