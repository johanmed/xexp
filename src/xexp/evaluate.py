"""Main script for evaluation"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RANDOM_SEED, collate_fn, gene_labels, test_dataset, tissue_labels
from nn import SetTransformer
from train import GELoss


def evaluate(model, dataloader, device="cuda") -> torch.Tensor:
    model = model.to(device)
    model.eval()
    metrics = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                obs_tissues=batch["obs_tissues"],
                obs_genes=batch["obs_genes"],
                obs_expressions=batch["obs_expressions"],
                query_tissues=batch["query_tissues"],
                query_genes=batch["query_genes"],
                obs_mask=batch["obs_mask"],
            )
            criterion = GELoss()
            losses = criterion(outputs, batch["targets"])
            metric = losses["total"].item() / len(batch)
            metrics.append(metric)
    return sum(metrics) / len(metrics)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = SetTransformer(
        n_tissues=len(np.unique(tissue_labels)),
        n_genes=len(np.unique(gene_labels)),
        dims=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
    )

    loaded_weights = torch.load("../../results/xexp_weights.pt", weights_only=True)
    model.load_state_dict(loaded_weights)

    print("Evaluation...")
    final_metric = evaluate(model, test_dataloader)
    print(f"Final metric: {final_metric}")
