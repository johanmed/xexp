"""Module with constructs for representing, processing and modeling gene expression data"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
                "tissues": torch.tensor(
                    [o.tissue for o in obs_list], dtype=torch.long
                ).squeeze(),
                "genes": torch.tensor(
                    [o.gene for o in obs_list], dtype=torch.long
                ).squeeze(),
                "expressions": torch.tensor(
                    [o.expression for o in obs_list], dtype=torch.float32
                ).squeeze(),
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
        self.n_samples = expression_matrix.shape[0]
        self.n_tissues = len(np.unique(tissue_labels))
        self.n_genes = len(np.unique(gene_labels))
        self.tissues = np.asarray(tissue_labels).flatten()
        self.genes = np.asarray(gene_labels).flatten()
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
        target_tissue = np.random.choice(query_tissue_ids)

        observed = []
        query = []

        for new_idx in range(len(sample)):
            t_idx = self.tissues[new_idx]
            g_idx = self.genes[new_idx]
            expression = sample[new_idx]
            if t_idx in obs_tissue_ids:
                observed.append(
                    GeneExpression(tissue=t_idx, gene=g_idx, expression=expression)
                )
            if t_idx == target_tissue:
                query.append(
                    GeneExpression(tissue=t_idx, gene=g_idx, expression=expression)
                )
        return Example(observations=observed, query=query)


def collate_fn(batch: list[Example]) -> dict:
    """
    Batch multiple Example into tensor dictionaries
    Handle padding for variable-length observations
    """
    tensor_dicts = [ex.to_tensor_dict() for ex in batch]

    max_obs_len = max(d["obs_tissues"].shape[0] for d in tensor_dicts)
    max_query_len = max(d["query_tissues"].shape[0] for d in tensor_dicts)

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
        obs_len = d["obs_tissues"].shape[0]
        query_len = d["query_tissues"].shape[0]

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


RANDOM_SEED = 2026
BATCH_SIZE = 2  # might need finetuning
N_SAMPLES = 100  # same

df = pd.read_csv(
    "../../data/xexp_data_sample.csv",
    names=["tissue", "gene", "expression"],
)
df = df.groupby(["tissue", "gene"]).head(N_SAMPLES).reset_index(drop=True)
df["sample_id"] = df.groupby(["tissue", "gene"]).cumcount()
min_samples = df.groupby(["tissue", "gene"]).size().min()
df = df[df["sample_id"] < min_samples]
print(f"Preparing dataset...\nDimensions: {df.shape}")

pivot = df.pivot_table(
    index="sample_id",
    columns=["tissue", "gene"],
    values="expression",
)

tissue_labels = pivot.columns.get_level_values(0).values.to_numpy().reshape(-1, 1)
gene_labels = pivot.columns.get_level_values(1).values.to_numpy().reshape(-1, 1)
expression_matrix = pivot.values
expression_scaler = StandardScaler()
expression_matrix = expression_scaler.fit_transform(expression_matrix)

tv_expression_matrix, test_expression_matrix = train_test_split(
    expression_matrix, test_size=0.2, random_state=RANDOM_SEED
)
train_expression_matrix, valid_expression_matrix = train_test_split(
    tv_expression_matrix, test_size=0.2, random_state=RANDOM_SEED
)

tissue_encoder, gene_encoder = LabelEncoder(), LabelEncoder()
tissue_labels = tissue_encoder.fit_transform(tissue_labels)
gene_labels = gene_encoder.fit_transform(gene_labels)

if (
    not Path("../../results/expression_scaler.pkl").exists()
    or not Path("../../results/tissue_encoder.pkl").exists()
    or not Path("../../results/gene_encoder.pkl").exists()
):
    joblib.dump(expression_scaler, "../../results/expression_scaler.pkl")
    joblib.dump(tissue_encoder, "../../results/tissue_encoder.pkl")
    joblib.dump(gene_encoder, "../../results/gene_encoder.pkl")

train_dataset = MicroarrayDataset(
    expression_matrix=train_expression_matrix,
    tissue_labels=tissue_labels,
    gene_labels=gene_labels,
)
valid_dataset = MicroarrayDataset(
    expression_matrix=valid_expression_matrix,
    tissue_labels=tissue_labels,
    gene_labels=gene_labels,
)
test_dataset = MicroarrayDataset(
    expression_matrix=test_expression_matrix,
    tissue_labels=tissue_labels,
    gene_labels=gene_labels,
)
