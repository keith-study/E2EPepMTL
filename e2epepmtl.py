import os
import time
import json
from typing import Dict, List, Union
import pandas as pd
import requests
from io import StringIO
from hestia.similarity import sequence_similarity_mmseqs
from hestia.partition import ccpart
from .data.residues import is_canonical
import torch
import esm
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import joblib

from .model.pepmtl import PepMTL
from .model.pepstl import PepSTL
from .model.pepstlcnn import PepSTL_CNN
from .model.pepstlbilstm import PepSTL_BiLSTM

os.environ["OPTUNA_DISABLE_DISTRIBUTED"] = "1"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

import optuna
from optuna.samplers import RandomSampler, TPESampler
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

import optuna
from optuna.samplers import RandomSampler, TPESampler
from sklearn.model_selection import train_test_split

import torch.nn.functional as F

from sklearn.model_selection import KFold

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import threading

import joblib
joblib.parallel.BACKENDS['multiprocessing'] = joblib.parallel.BACKENDS['loky']

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split

_global_esm_cache = {}
__version__ = "0.2.0"

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

def _get_cached_esm_model(model_name: str):
    """Retrieve or initialize a pretrained ESM-2 model from a global cache.

    This ensures that heavy model weights (ESM-2) and their associated alphabets
    are only loaded into memory once per session, significantly speeding up
    subsequent embedding extractions.

    Args:
        model_name: Name of the ESM-2 model (e.g., 'esm2_t33_650M_UR50D')

    Returns:
        tuple: (model, alphabet, device)
            - model: The loaded torch.nn.Module in eval mode
            - alphabet: The ESM vocabulary and tokenizer helper
            - device: The torch.device (CUDA or CPU) the model is residing on
    """

    if model_name not in _global_esm_cache:

        print(f"\n[ESM Cache] First-time loading model: {model_name} "
              f"(this will take 30-60s, subsequent calls will be instant)...")

        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"[Parallel] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
            model = torch.nn.DataParallel(model)

        model = model.to(device)

        # Store in global cache
        _global_esm_cache[model_name] = (model, alphabet, device)
        torch.cuda.empty_cache()
        print(f"[ESM Cache] Model {model_name} loaded and cached successfully!\n")

    return _global_esm_cache[model_name]


import torch
import torch.nn as nn
import torch.nn.functional as F


# class UncertaintyLoss(nn.Module):
#     def __init__(self, categories, custom_weights=None):
#         super().__init__()
#
#         if len(set(categories)) != len(categories):
#             duplicates = [item for item in set(categories) if categories.count(item) > 1]
#             raise ValueError(
#                 f"The categories list contains duplicate items: {duplicates}. "
#                 f"Expected {len(categories)} unique items, got {len(set(categories))}."
#             )
#
#         self.num_tasks = len(categories)
#         self.categories = categories
#         self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
#
#         weights = torch.ones(self.num_tasks)
#         if custom_weights:
#             for task_name, multiplier in custom_weights.items():
#                 if task_name in categories:
#                     idx = categories.index(task_name)
#                     weights[idx] = multiplier
#                     print(f"介入成功: {task_name} (Index {idx}) 權重設為 {multiplier}x")
#
#         self.register_buffer('task_importance_weights', weights)
#
#         # 新增：定義要套用 label smoothing 的任務索引
#         self.smoothing_tasks = ['QS', 'TOX', 'ACE']   # 你想針對的任務名稱
#         self.smoothing = 0.12                         # 平滑強度，可調 0.1~0.15
#         self.smoothing_indices = [
#             self.categories.index(task) for task in self.smoothing_tasks
#             if task in self.categories
#         ]
#         print(f"Label Smoothing 將套用於任務: {self.smoothing_tasks} (indices: {self.smoothing_indices})")
#
#     def forward(self, preds, targets, pos_weights):
#         total_loss = 0
#
#         for i in range(self.num_tasks):
#             # 原始 target
#             target = targets[:, i]
#
#             # 如果這個任務需要 smoothing
#             if i in self.smoothing_indices:
#                 # 正樣本：1.0 → 1 - smoothing/2 ≈ 0.94
#                 # 負樣本：0.0 → smoothing/2 ≈ 0.06
#                 target = target * (1 - self.smoothing) + (self.smoothing / 2)
#
#             # 計算 BCE loss（使用平滑後的 target）
#             task_loss = F.binary_cross_entropy_with_logits(
#                 preds[:, i],
#                 target,
#                 pos_weight=pos_weights[i]
#             )
#
#             # Uncertainty weighting（不變）
#             precision = torch.exp(-self.log_vars[i])
#             weighted_task_loss = (precision * task_loss * self.task_importance_weights[i]) + self.log_vars[i]
#
#             total_loss += weighted_task_loss
#
#         return total_loss

# 雖然 GatedTaskHead 建議放 pepmtl.py，但若你想集中管理，加在這裡也可以
class GatedTaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.2):
        super(GatedTaskHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一層
        residual = self.fc1(x)
        x = self.relu(self.bn1(residual))
        x = self.dropout(x)

        # 第二層 (殘差連接)
        x = self.fc2(x)
        x = self.relu(self.bn2(x + residual))  # 讓特定任務能保留更多 backbone 資訊

        return self.fc3(x)



class E2EPepMTL:
    """
        E2EPepMTL: End-to-End Peptide Multi-Task Learning Framework
        A complete automated framework for peptide bioactivity prediction.
        """
    def __init__(self,
                 verbose: bool = True,
                 threads: int = 8,
                 seed: int = 42
                 ):
        self.verbose = verbose
        self.threads = threads
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.categories = [
            'AB', 'ACE', 'ACP', 'AF', 'AMAP', 'AMP', 'AOX', 'APP',
            'AV','BBP', 'DPPIV', 'MRSA', 'NEURO', 'QS', 'TOX', 'TTCA'
        ]
        self.db = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data', 'peptipedia'
        )
        os.makedirs(self.db, exist_ok=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.verbose:
            self.welcome()

    def welcome(self) -> None:
        """Display welcome message"""
        if self.verbose:
            message = '| Welcome to E2EPepMTL |'
            print('-' * len(message))
            print(message)
            print('-' * len(message))
            print("An End-to-End Automated Framework for for Multi-Task Peptide Property Prediction")
            print(f"Version: {__version__}")
            print("=" * 80)

    def curate_dataset(
            self,
            positives_dir: str,
            outputdir: str = None
    ) -> pd.DataFrame:
        """
        Load and merge category-specific CSVs into a single multi-label DataFrame.
        Args:
            positives_dir: Directory containing CSV files named as '{category}_positives.csv'
            outputdir: Optional directory to save the consolidated 'merged_positives.csv'
        Returns:
            pd.DataFrame: A DataFrame with a unique 'sequence' column and binary indicator
                          columns (Y_AB, Y_AMP, etc.) for each category
        Raises:
            ValueError: If a required category file is missing or no sequence column is detected
        """

        if self.verbose:
            print(f"Merging positive datasets from directory")

        # Initialize an empty DataFrame
        df = pd.DataFrame()

        for cat in self.categories:
            # Path validation for each bioactivity category
            csv_path = os.path.join(positives_dir, f"{cat}_positives.csv")
            if not os.path.exists(csv_path):
                raise ValueError(f"Missing file: {csv_path}")

            df_cat = pd.read_csv(csv_path)

            # Handle varying naming conventions across datasets
            seq_col = None
            possible_names = ['sequence', 'seq', 'peptide', 'peptide_sequence', 'seq_id']
            for col in df_cat.columns:
                if col.lower() in possible_names:
                    seq_col = col
                    break
            if seq_col is None:
                raise ValueError(
                    f"Cannot find a sequence column in {csv_path}. "
                    f"Columns: {list(df_cat.columns)}"
                )
            # Ensure consistency for merging
            df_cat['sequence'] = (
                df_cat[seq_col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            # Remove duplicates and non-canonical amino acids
            df_cat.drop_duplicates('sequence', inplace=True)
            df_cat = df_cat[df_cat['sequence'].map(is_canonical)].reset_index(drop=True)
            df_cat = df_cat[~df_cat['sequence'].isna()]

            # Mark sequences in this file as positive for the current category
            df_cat[f'Y_{cat}'] = 1

            # Handle overlapping bioactivities
            if df.empty:
                df = df_cat[['sequence', f'Y_{cat}']]
            else:
                # If a peptide has multiple functions, they are merged into one row here
                df = pd.merge(df, df_cat[['sequence', f'Y_{cat}']], on='sequence', how='outer')

        # Fill missing values with 0
        for cat in self.categories:
            col = f'Y_{cat}'
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0).astype(int)      # Ensure binary integer format

        # Statistical Reporting
        if self.verbose:
            print("\nMerged dataset statistics:")
            print(f"- Unique sequences : {len(df)}")
            for cat in self.categories:
                print(f"- Positive for {cat:5}: {int(df[f'Y_{cat}'].sum())}")

        # Save the merged dataset
        if outputdir:
            os.makedirs(outputdir, exist_ok=True)
            merged_path = os.path.join(outputdir, "merged_positives.csv")
            df.to_csv(merged_path, index=False)
            if self.verbose:
                print(f"Merged file saved to: {merged_path}")

        return df

    def autosearch_negatives(
            self,
            df_pos: pd.DataFrame,
            positive_tags: List[str],
            proportion: float = 1.0,
            save: bool = False,
            autosearch: str = 'auto',
            outputdir: str = None
    ) -> pd.DataFrame:
        """
        Fetch negative samples with controlled source priority and length distribution matching
        Args:
            df_pos: Positive samples DataFrame containing active sequences
            positive_tags: Bioactivity tags to exclude from the negative pool
            proportion: Target ratio of Negative:Positive samples (default 1.0)
            save: Whether to save downloaded FASTA and processed CSVs to disk
            autosearch: Search mode:
                        'auto'/'true': Try [Peptipedia API](https://api.app.peptipedia.cl) first, then local
                        'false': Skip API, use local [FASTA] only.
            outputdir: Directory path to save generated negative sample files
        Returns:
            pd.DataFrame: A shuffled dataset combining positive and distribution-matched negative samples
        """

        global start_time
        if self.verbose:
            print("Searching for negative samples...")
            start_time = time.time()

        # If no negatives are requested, return original positives
        if proportion <= 0:
            if self.verbose:
                print("Dataset already contains negative samples, skipping search")
            return df_pos

        # Attempt to fetch the latest peptide dataset from Peptipedia (only if autosearch is not 'false')
        if autosearch != 'false':
            try:
                if self.verbose:
                    print("Attempting to download from Peptipedia API...")

                api_url = "https://api.app.peptipedia.cl/files/downloads/all_peptides.fasta"
                response = requests.get(api_url, timeout=30)
                response.raise_for_status()

                # Cache the file locally for future offline use
                fasta_path = os.path.join(self.db, "peptipedia_all_peptides.fasta")
                with open(fasta_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)

                if self.verbose:
                    print(f"Successfully downloaded and saved FASTA to: {fasta_path}")

                # Process the downloaded dataset into a negative dataset
                df_all = self.parse_fasta_to_df(response.text)
                df_neg = self.process_negative_samples(df_all, df_pos, positive_tags, proportion)

                # Save intermediate results if requested
                if save:
                    csv_path = os.path.join(self.db, "peptipedia_all_peptides.csv")
                    df_all.to_csv(csv_path, index=False)
                    neg_csv_path = os.path.join(outputdir, "negative_samples.csv")
                    df_neg.to_csv(neg_csv_path, index=False)
                    if self.verbose:
                        print("Saved intermediate CSV files")

                df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=self.seed)

                return df

            except Exception as e:
                if self.verbose:
                    print(f"\nAPI request failed: ({str(e)}), falling back to local peptide database")

        # Process the existing FASTA file (for both autosearch=false and API fail cases)
        fasta_path = os.path.join(self.db, "peptipedia_all_peptides.fasta")

        if os.path.exists(fasta_path):
            try:
                if self.verbose:
                    print("Found local FASTA file, processing with length binning...")

                with open(fasta_path, 'r', encoding='utf-8') as f:
                    fasta_content = f.read()

                df_all = self.parse_fasta_to_df(fasta_content)
                df_neg, df_full = self.process_negative_samples(df_all, df_pos, positive_tags, proportion)

                if save and outputdir:
                    df_all.to_csv(os.path.join(self.db, "peptipedia_all_peptides.csv"), index=False)
                    df_neg.to_csv(os.path.join(outputdir, "negative_samples.csv"), index=False)
                    if self.verbose:
                        print(f"Negative sample saved to: {os.path.join(outputdir, 'negative_samples.csv')}")

                return df_full

            except Exception as e:
                if self.verbose:
                    print(f"Error processing local FASTA: {str(e)}")
        return df

    def process_negative_samples(
            self,
            df_all: pd.DataFrame,
            df_pos: pd.DataFrame,
            positive_tags: List[str],
            proportion: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process negative samples with length binning approach
        Args:
            df_all: Raw peptide pool from API or local FASTA
            df_pos: Current positive sample DataFrame
            positive_tags: Bioactivity tags to exclude (ensuring non-overlap)
            proportion: Desired Negative:Positive ratio (e.g., 1.0 for balanced)
        Returns:
            tuple: (df_result, df_full)
                - df_result: Only the newly generated negative samples
                - df_full: Combined, labeled, and shuffled positive + negative dataset
        """

        # Filter sequences by length range [MIN_LEN, MAX_LEN]
        MIN_LEN, MAX_LEN = 5, 150
        df_all = df_all[df_all['sequence'].str.len().between(MIN_LEN, MAX_LEN)].copy()
        df_pos_filtered = df_pos[df_pos['sequence'].str.len().between(MIN_LEN, MAX_LEN)].copy()

        if self.verbose:
            print(f"\nLength filtering [{MIN_LEN}, {MAX_LEN}]: "
                  f"Positive: {len(df_pos_filtered)}, Peptipedia: {len(df_all)}")

        # Keep only canonical amino acids and remove redundancy
        df_all = df_all[df_all['sequence'].map(is_canonical)].copy()
        df_all.drop_duplicates('sequence', inplace=True)

        # Ensure negative dataset does not contain any positive sequences
        pos_seqs = set(df_pos_filtered['sequence'])
        df_neg_raw = df_all[~df_all['sequence'].isin(pos_seqs)].copy()

        # Handle no valid negative samples found
        if df_neg_raw.empty:
            empty_neg = pd.DataFrame(columns=['sequence'])
            for col in [f'Y_{c}' for c in self.categories]:
                empty_neg[col] = 0
            empty_neg['Y'] = 0
            return empty_neg, df_pos_filtered.assign(Y=1)

        # Create 5-residue bins to match length distribution
        df_neg_raw['length'] = df_neg_raw['sequence'].str.len()
        df_pos_filtered['length'] = df_pos_filtered['sequence'].str.len()

        min_len = int(df_pos_filtered['length'].min())
        max_len = int(df_pos_filtered['length'].max())

        # Define bin edges (e.g., [5, 10, 15...])
        bin_edges = list(range((min_len // 5) * 5, (max_len // 5 + 2) * 5, 5))
        bins = pd.interval_range(start=bin_edges[0], end=bin_edges[-1], freq=5, closed='left')

        # Assign each sequence to a bin
        df_neg_raw['bin'] = pd.cut(df_neg_raw['length'], bins=bins)
        df_pos_filtered['bin'] = pd.cut(df_pos_filtered['length'], bins=bins)

        # Calculate how many negatives need for each length interval
        pos_bin_counts = df_pos_filtered['bin'].value_counts()
        target_per_bin = (pos_bin_counts * proportion).astype(int)

        # Proportional Sampling with Forward Compensation
        # If one bin lacks enough samples, 'carry over' the deficit to the next bin
        samples = []
        cumulative_missing = 0
        sorted_bins = sorted(target_per_bin.index, key=lambda x: x.left if pd.notna(x) else float('inf'))

        for bin_int in sorted_bins:
            if pd.isna(bin_int):
                continue
            target = target_per_bin[bin_int]
            pool = df_neg_raw[df_neg_raw['bin'] == bin_int]
            pool_size = len(pool)
            need = target + cumulative_missing

            if pool_size == 0:
                cumulative_missing += target
                continue

            take = min(need, pool_size)
            sampled = pool.sample(n=take, random_state=self.seed)
            samples.append(sampled[['sequence']])
            cumulative_missing = max(0, need - take)

        df_result = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=['sequence'])

        # Generate 0 columns for negative samples
        y_cols = [f'Y_{cat}' for cat in self.categories]
        for col in y_cols:
            df_result[col] = 0

        # Prepare positive samples for merging
        df_pos_clean = df_pos_filtered[['sequence'] + y_cols].copy()

        # Merge and shuffle the complete dataset
        df_full = pd.concat([df_pos_clean, df_result], ignore_index=True)
        df_full = df_full.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        if self.verbose:
            final_missing = len(df_pos_filtered) - len(df_result)
            print(f"\nNegative sample search completed (Time: {time.time() - start_time:.2f}s)")
            # print(f"- Target negative sample: {len(df_pos_filtered)}")
            # print(f"- Actual negative sample: {len(df_result)}")
            # print(f"- Final missing: {final_missing}")

        # overlap = set(df_neg_raw['sequence']) & set(df_pos_filtered['sequence'])
        # if len(overlap) == 0:
        #     print("No overlap between positive and negative samples")
        # else:
        #     print(f"Alert: {len(overlap)} sequence(s) overlap!")

        return df_result, df_full

    def parse_fasta_to_df(
            self,
            fasta_content: str
    ) -> pd.DataFrame:
        """
        Parse FASTA content into a structured DataFrame.
        Args:
            fasta_content: A string containing the full FASTA file content

        Returns:
            pd.DataFrame: A DataFrame with columns ['id', 'sequence', 'is_canonical']
        """

        records = []
        current_id = ""

        # Use StringIO to treat the string as a file for line-by-line memory efficiency
        for line in StringIO(fasta_content):
            line = line.strip()
            # Extract the metadata line starting with '>'
            if line.startswith(">"):
                current_id = line[1:]
            # Append sequence data linked to the current header
            elif line:
                records.append({
                    "id": current_id,
                    "sequence": line,
                    "is_canonical": is_canonical(line)
                })

        # Convert the list of dictionaries into a DataFrame for downstream processing
        return pd.DataFrame(records)

    def homology_partitioning(
            self,
            df: pd.DataFrame,
            test_size: float = 0.2,
            similarity_threshold: float = 0.3,
            sequence_col: str = 'sequence',
            outputdir: str = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into Train/Test sets while strictly preventing data leakage due to sequence homology
        Args:
            df: Input DataFrame containing peptide sequences.
            test_size: Target proportion for the test set (e.g., 0.2)
            similarity_threshold: Identity threshold (0.0 to 1.0). Sequences above
                                  this are considered homologous
            sequence_col: Column name for the peptide strings
            outputdir: Directory to save the partitioned CSV files
        Returns:
            tuple: (df_train, df_test) - The homology-separated DataFrames
        """

        # Ensure data integrity
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if sequence_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{sequence_col}' column")

        if self.verbose:
            print(f"- Using similarity threshold: {similarity_threshold}")
            print(f"- Target test size: {test_size * 100}% of data")
            print(f"- Input DataFrame shape: {df.shape}")

        # Reset indices for clean indexing
        df = df.reset_index(drop=True)

        try:
            # Execute MMseqs2 to find pairwise identities
            if self.verbose:
                print("\nCalculating pairwise sequence similarities with MMseqs2...")
                start_time = time.time()

            # Get similarity matrix
            sim_df = sequence_similarity_mmseqs(
                df,
                field_name=sequence_col,
                prefilter=True
            )

            if self.verbose:
                print(f"Similarity calculation completed in {time.time() - start_time:.2f}s")
                print(f"- Similarity matrix shape: {sim_df.shape}")

            # Apply CCPart to assign indices to train or test buckets
            if self.verbose:
                print("\nApplying CCPart algorithm...")

            # train_indices and test_indices are chosen such that cross-set similarity < threshold
            train_indices, test_indices, _ = ccpart(
                df=df,
                threshold=similarity_threshold,
                test_size=test_size,
                sim_df=sim_df,
            )

            # Extract subsets based on calculated indices
            df_train = df.iloc[train_indices].copy()
            df_test = df.iloc[test_indices].copy()

            if self.verbose:
                print("Partitioning completed:")
                print(f"- Training set size: {len(df_train)}")
                print(f"- Test set size: {len(df_test)}")
                print(f"- Actual test ratio: {len(df_test) / len(df):.2f}")

            # Save partitioned datasets
            if outputdir:
                os.makedirs(outputdir, exist_ok=True)
                train_path = os.path.join(outputdir, "train_dataset.csv")
                test_path = os.path.join(outputdir, "test_dataset.csv")
                df_train.to_csv(train_path, index=False)
                df_test.to_csv(test_path, index=False)
                if self.verbose:
                    print(f"- Saved train_dataset.csv to: {train_path}")
                    print(f"- Saved test_dataset.csv to: {test_path}")

            return df_train, df_test

        except Exception as e:
            # Re-raise with context
            error_msg = f"Homology partitioning failed: {str(e)}"
            if self.verbose:
                print(error_msg)
            raise RuntimeError(error_msg) from e

    def generate_esm_embeddings(
            self,
            df: pd.DataFrame,
            sequence_col: str = 'sequence',
            model_name: str = 'esm2_t33_650M_UR50D',
            batch_size: int = 1024,
            outputdir: str = None,
    ) -> np.ndarray:
        """
        Extract high-dimensional features from sequences using pretrained ESM-2 models
        Args:
            df: Input DataFrame containing peptide sequences.
            sequence_col: Column name for the peptide strings.
            model_name: ESM-2 model variant (e.g., 650M or 3B parameters).
            batch_size: Number of sequences to process per GPU inference cycle.
            outputdir: Path to save the resulting .npy embedding matrix.
        Returns:
            np.ndarray: Matrix of shape (n_samples, embedding_dim).
        """

        if self.verbose:
            print("Generating ESM-2 embeddings...")

        start_time = time.time()

        # Fetch model/alphabet from the optimized global cache
        model, alphabet, device = _get_cached_esm_model(model_name)
        batch_converter = alphabet.get_batch_converter()

        # Format data for the ESM tokenizer: [(label, sequence), ...]
        sequences = [(f"seq_{i}", row[sequence_col]) for i, row in df.iterrows()]
        embeddings = []

        # Maximize GPU utilization
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)

            # Calculate true sequence lengths (excluding padding)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                # Identify the target Transformer layer for extraction
                layer_map = {
                    "t6": 6, "t8": 8, "t12": 12, "t30": 30,
                    "t33": 33, "t36": 36, "t40": 40, "t48": 48
                }
                layer = next((v for k, v in layer_map.items() if k in model_name), 33)
                # Extract internal hidden states
                results = model(batch_tokens, repr_layers=[layer], return_contacts=False)

            token_reps = results["representations"][layer]

            # Slicing [1:len-1] to remove <cls> and <eos> tokens
            for j, tokens_len in enumerate(batch_lens):
                # Calculate the mean of residue tokens to get a "global" peptide vector
                seq_rep = token_reps[j, 1:tokens_len - 1].mean(0).cpu().numpy()
                embeddings.append(seq_rep)

            del batch_tokens, results, token_reps
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        embeddings = np.array(embeddings)

        if self.verbose:
            print(f"- Embeddings generated: shape {embeddings.shape}, time {time.time() - start_time:.2f}s")

        # Save the binary matrix for downstream ML training
        if outputdir:
            safe_name = model_name.replace('/', '_')
            path = os.path.join(outputdir, f"esm_embeddings_{safe_name}.npy")
            np.save(path, embeddings)
            if self.verbose:
                print(f"- Embeddings saved to: {path}")

        return embeddings

    def prepare_benchmark_dataset(
            self,
            benchmark_dir: str,
            model_name: str = "esm2_t33_650M_UR50D",
            batch_size: int = 1024,
            outputdir: str = None,
            reuse_embeddings: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare benchmark dataset by merging all single-task train/test splits.
        This function merges 32 CSV files (16 tasks × train/test) using 'sequence' as the key
        to eliminate duplicate sequences while preserving all task labels. Sequences appearing
        in multiple tasks are kept as a single row with multiple Y_XXX columns.
        Process:
            1. Load all per-task CSVs
            2. Merge on 'sequence' using outer join
            3. Fill missing labels with 0 (negative)
            4. Filter out sequences with no positive labels
            5. Generate or reuse ESM-2 embeddings
        Args:
            benchmark_dir: Directory containing *_train.csv and *_test.csv files.
            model_name: ESM-2 model variant to use for embeddings.
            batch_size: Batch size for ESM-2 inference.
            outputdir: Directory to save/load embeddings (default: benchmark_dir).
            reuse_embeddings: Whether to load existing embeddings if available.
        Returns:
            tuple: (df_train, df_test, train_embeddings, test_embeddings)
    """

        print(f"\n=== Processing Benchmark Dataset ({len(self.categories)} single-task splits) ===")
        print(f"Directory: {benchmark_dir}")

        # Initialize empty DataFrames with 'sequence' column as key
        df_train = pd.DataFrame(columns=['sequence'])
        df_test = pd.DataFrame(columns=['sequence'])

        # Counters for raw sample counts (before any cleaning or merging)
        total_raw_train = 0
        total_raw_test = 0

        for cat in self.categories:
            train_path = os.path.join(benchmark_dir, f"{cat}_train.csv")
            test_path = os.path.join(benchmark_dir, f"{cat}_test.csv")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"Missing file: {train_path} or {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Print raw sample count for this task (before any cleaning)
            raw_train_count = len(train_df)
            raw_test_count = len(test_df)
            total_raw_train += raw_train_count
            total_raw_test += raw_test_count

            print(f"Read {cat}_train.csv → Raw sample count: {raw_train_count:,}")
            print(f"Read {cat}_test.csv  → Raw sample count: {raw_test_count:,}")

            # Find sequence and label columns (support multiple naming conventions)
            seq_col = next((c for c in train_df.columns if c.lower() in ['sequence', 'seq', 'peptide']), None)
            label_col = next((c for c in train_df.columns if c == 'Y'),
                             None)  # Your label column is 'Y' (case-sensitive)

            if seq_col is None or label_col is None:
                raise ValueError(f"{cat}_train.csv missing 'sequence' or 'Y' column "
                                 f"(detected columns: {train_df.columns.tolist()})")

            # Keep only sequence and label columns, rename label to Y_{cat}
            train_df = train_df[[seq_col, label_col]].rename(columns={seq_col: 'sequence', label_col: f'Y_{cat}'})
            test_df = test_df[[seq_col, label_col]].rename(columns={seq_col: 'sequence', label_col: f'Y_{cat}'})

            # Clean sequences: convert to string, remove leading/trailing spaces, convert to uppercase
            train_df['sequence'] = train_df['sequence'].astype(str).str.strip().str.upper()
            test_df['sequence'] = test_df['sequence'].astype(str).str.strip().str.upper()

            # Merge using 'sequence' as key (outer join to keep all unique sequences)
            df_train = df_train.merge(train_df, on='sequence', how='outer')
            df_test = df_test.merge(test_df, on='sequence', how='outer')

        # Print total raw counts (before cleaning and merging)
        print(f"\nAll tasks loaded (before cleaning and merging):")
        print(f"  - Total raw train samples (sum of all _train.csv): {total_raw_train:,}")
        print(f"  - Total raw test samples (sum of all _test.csv):  {total_raw_test:,}")

        # Fill missing task labels with 0 (sequences not labeled in a task are negative for that task)
        for cat in self.categories:
            col = f'Y_{cat}'
            if col not in df_train.columns:
                df_train[col] = 0
            else:
                df_train[col] = df_train[col].fillna(0).astype(int)

            if col not in df_test.columns:
                df_test[col] = 0
            else:
                df_test[col] = df_test[col].fillna(0).astype(int)

        # Filter out sequences with no positive labels in any task (optional but recommended)
        y_cols = [f'Y_{cat}' for cat in self.categories]
        df_train = df_train[df_train[y_cols].sum(axis=1) > 0].reset_index(drop=True)
        df_test = df_test[df_test[y_cols].sum(axis=1) > 0].reset_index(drop=True)

        # Final merged statistics
        print(f"Merged completed (duplicates removed via sequence key):")
        print(f"  - Train sample count: {len(df_train):,}")
        print(f"  - Test sample count: {len(df_test):,}")
        print(f"  - Total positive labels: {int(df_train[y_cols].sum().sum()):,}")

        # Check for multi-functional peptides (sequences positive in multiple tasks)
        multi_func = df_train[y_cols].sum(axis=1) > 1
        if multi_func.any():
            print(f"Found {multi_func.sum()} multi-functional peptides (positive in multiple tasks)")

        # Generate or load ESM-2 embeddings
        emb_dir = outputdir or benchmark_dir
        os.makedirs(emb_dir, exist_ok=True)

        train_emb_path = os.path.join(emb_dir, "benchmark_train_embeddings.npy")
        test_emb_path = os.path.join(emb_dir, "benchmark_test_embeddings.npy")

        if reuse_embeddings and os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
            print("Loading existing benchmark embeddings...")
            train_emb = np.load(train_emb_path)
            test_emb = np.load(test_emb_path)
        else:
            print("Computing ESM-2 embeddings...")
            train_emb = self.generate_esm_embeddings(
                df_train, sequence_col='sequence',
                model_name=model_name, batch_size=batch_size, outputdir=emb_dir
            )
            test_emb = self.generate_esm_embeddings(
                df_test, sequence_col='sequence',
                model_name=model_name, batch_size=batch_size, outputdir=emb_dir
            )
            np.save(train_emb_path, train_emb)
            np.save(test_emb_path, test_emb)
            print(f"Embeddings saved to: {train_emb_path}, {test_emb_path}")

        # Shape consistency check
        assert train_emb.shape[0] == len(df_train), "train_emb shape mismatch with df_train"
        assert test_emb.shape[0] == len(df_test), "test_emb shape mismatch with df_test"

        print("Benchmark data preparation completed, ready for training.")
        return df_train, df_test, train_emb, test_emb

    def create_model(
            self,
            hparams: dict,
            input_dim: int,
            num_tasks: int
    ):
        """
        Create and initialize the PepMTL model with given hyperparameters.
        Args:
            hparams: Dictionary containing model hyperparameters.
            input_dim: Dimension of the input features (e.g. ESM embedding size).
            num_tasks: Number of prediction tasks (multi-task learning).
        Returns:
            torch.nn.Module: The initialized model (possibly wrapped with DataParallel).
        """
        model = PepMTL(
            input_dim=input_dim,
            num_tasks=num_tasks,
            lstm_num_layers=hparams.get('lstm_num_layers', 2),
            lstm_hidden=hparams.get('lstm_hidden', 512),
            cnn_out=hparams.get('cnn_out', 384),
            cnn_kernel=hparams.get('cnn_kernel', 3),
            head_hidden=hparams.get('head_hidden', 512),
            dropout=hparams.get('dropout', 0.3),
        )

        # Enable multi-GPU training if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(self.device)

        return model

    def create_model_stl(
            self,
            hparams,
            input_dim,
            backbone_type="hybrid"
    ):
        """
        Create a Single-Task Learning (STL) model with different backbone options.
        Args:
            hparams: Hyperparameter dictionary.
            input_dim: Dimension of input features (e.g., ESM embedding size).
            backbone_type: Type of backbone architecture. Options: "hybrid", "cnn", "bilstm".
        Returns:
            torch.nn.Module: Initialized PepSTL model variant.
        Raises:
            ValueError: If unknown backbone_type is provided.
        """

        if backbone_type == "hybrid":
            return PepSTL(
                input_dim=input_dim,
                lstm_num_layers=hparams.get('lstm_num_layers', 2),
                lstm_hidden=hparams.get('lstm_hidden', 512),
                cnn_out=hparams.get('cnn_out', 256),
                cnn_kernel=hparams.get('cnn_kernel', 3),
                head_hidden=hparams.get('head_hidden', 512),
                dropout=hparams.get('dropout', 0.3)
            )
        elif backbone_type == "cnn":
            return PepSTL_CNN(
                input_dim=input_dim,
                cnn_out=hparams.get('cnn_out', 256),
                cnn_kernel=hparams.get('cnn_kernel', 3),
                head_hidden=hparams.get('head_hidden', 512),
                dropout=hparams.get('dropout', 0.3)
            )
        elif backbone_type == "bilstm":
            return PepSTL_BiLSTM(
                input_dim=input_dim,
                lstm_num_layers=hparams.get('lstm_num_layers', 2),
                lstm_hidden=hparams.get('lstm_hidden', 512),
                head_hidden=hparams.get('head_hidden', 512),
                dropout=hparams.get('dropout', 0.3)
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

    def create_model_bilstm(
            self,
            hparams: dict,
            input_dim: int,
            num_tasks: int
    ):
        """
        Create a pure BiLSTM-based multi-task learning model.
        Args:
            hparams: Hyperparameter dictionary containing:
                     - lstm_hidden, lstm_num_layers, head_hidden, dropout
            input_dim: Dimension of input features (e.g. ESM embedding dim).
            num_tasks: Number of prediction tasks.
        Returns:
            torch.nn.Module: Initialized PureBiLSTMMTL model.
        """

        class PureBiLSTMMTL(nn.Module):
            """Pure BiLSTM model for multi-task peptide prediction."""
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hparams['lstm_hidden'],
                    num_layers=hparams['lstm_num_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=hparams['dropout'] if hparams['lstm_num_layers'] > 1 else 0
                )
                lstm_out_dim = hparams['lstm_hidden'] * 2   # bidirectional

                self.head = nn.Sequential(
                    nn.Linear(lstm_out_dim, hparams['head_hidden']),
                    nn.ReLU(),
                    nn.Dropout(hparams['dropout']),
                    nn.Linear(hparams['head_hidden'], num_tasks)
                )

            def forward(self, x):
                # Add sequence dimension if input is (batch, dim)
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # (batch, 1, dim)

                lstm_out, _ = self.lstm(x)
                # Global average pooling
                pooled = lstm_out.mean(dim=1)

                out = self.head(pooled)
                return out, None

        return PureBiLSTMMTL().to(self.device)

    def create_model_cnn(
            self,
            hparams: dict,
            input_dim: int,
            num_tasks: int
    ):
        """
        Create a pure CNN-based multi-task learning model.
        Args:
            hparams: Hyperparameter dictionary containing:
                     - cnn_out, cnn_kernel, head_hidden, dropout
            input_dim: Dimension of input features (e.g. ESM embedding dim).
            num_tasks: Number of prediction tasks.
        Returns:
            torch.nn.Module: Initialized PureCNNMTL model.
        """

        class PureCNNMTL(nn.Module):
            def __init__(self):
                super().__init__()
                c = hparams.get('cnn_out', 256)
                k = hparams.get('cnn_kernel', 5)

                # 1D Convolution layers
                self.conv1 = nn.Conv1d(in_channels=1, out_channels=c, kernel_size=k, padding=k // 2)
                self.conv2 = nn.Conv1d(in_channels=c, out_channels=c * 2, kernel_size=k, padding=k // 2)
                self.pool = nn.AdaptiveAvgPool1d(1)
                cnn_dim = c * 2

                # Classification head
                self.head = nn.Sequential(
                    nn.Linear(cnn_dim, hparams['head_hidden']),
                    nn.ReLU(),
                    nn.Dropout(hparams['dropout']),
                    nn.Linear(hparams['head_hidden'], num_tasks)
                )

            def forward(self, x):
                # (batch, dim) -> (batch, channels, length)
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # (batch, 1, dim)

                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))

                # Global average pooling
                x = self.pool(x).squeeze(-1)

                return self.head(x), None   # Keep consistent output format

        model = PureCNNMTL().to(self.device)

        return model

    def evaluate_model(self, model, x_data, y_data, batch_size, device, num_tasks, return_per_task=False):
        """
        Evaluate model performance using Matthews Correlation Coefficient (MCC).
        Args:
            model: Trained model to evaluate.
            x_data: Input features (numpy array or torch tensor).
            y_data: Ground truth labels, shape (n_samples, num_tasks).
            batch_size: Batch size for inference.
            device: torch device (cuda/cpu).
            num_tasks: Number of tasks.
            return_per_task: Whether to return per-task MCC scores.
        Returns:
            float: Overall mean MCC (or tuple of (overall_mcc, task_mccs) if return_per_task=True)
        """
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(x_data), batch_size):
                slice_data = x_data[i:i + batch_size]

                # Convert numpy array to tensor and move to target device
                if isinstance(slice_data, np.ndarray):
                    batch = torch.tensor(slice_data, dtype=torch.float32).to(device)
                else:
                    batch = slice_data.to(device) if slice_data.device != device else slice_data

                task_pred, _ = model(batch)
                #task_pred = model(batch)
                preds.append(torch.sigmoid(task_pred).cpu())
                torch.cuda.empty_cache()
            pred = torch.cat(preds).numpy()

        # Binary prediction with default threshold 0.5
        bin_pred = (pred > 0.5).astype(int)

        task_mccs = []
        valid_mccs = []

        for j in range(num_tasks):
            true = y_data[:, j]
            # Skip tasks with no positive sample
            if true.sum() == 0:
                task_mccs.append(np.nan)
                continue

            best_task_mcc = -1.0
            # Search for the best threshold between 0.1 and 0.9 (step size 0.05)
            for th in np.arange(0.1, 0.9, 0.05):
                bin_pred_task = (pred[:, j] > th).astype(int)
                mcc = matthews_corrcoef(true, bin_pred_task)
                if mcc > best_task_mcc:
                    best_task_mcc = mcc

            best_task_mcc = max(0.0, best_task_mcc) if not np.isnan(best_task_mcc) else 0.0
            task_mccs.append(best_task_mcc)
            valid_mccs.append(best_task_mcc)

        overall_mcc = np.mean(valid_mccs) if valid_mccs else 0.0

        if return_per_task:
            return overall_mcc, task_mccs

        return overall_mcc

    def ensemble(self, best_params, x_train_np, y_train_np, x_test_np, y_test_np, outputdir, current_seed, num_tasks=16,
                 n_folds=10):
        """
        Perform K-fold cross-validation soft-voting ensemble training and evaluation.
        Args:
            best_params: Optimal hyperparameters for the model architecture.
            x_train_np: Training input features.
            y_train_np: Training ground truth labels, shape (n_samples, num_tasks).
            x_test_np: Test input features for evaluation.
            y_test_np: Test ground truth labels, shape (n_samples, num_tasks).
            outputdir: Directory path to save output predictions.
            current_seed: Random seed for reproducibility in KFold split.
            num_tasks: Number of prediction tasks. Defaults to 16.
            n_folds: Number of folds for cross-validation. Defaults to 10.
        Returns:
            tuple: (overall_mcc, ensemble_mccs, task_names, ensemble_prob)
                - overall_mcc (float): Mean MCC score across all valid tasks.
                - ensemble_mccs (list): Per-task MCC scores.
                - task_names (list): List of evaluated task category names.
                - ensemble_prob (np.ndarray): Ensembled prediction probabilities.
        """

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=current_seed)
        fold_probs = [None] * n_folds

        n_gpus = torch.cuda.device_count()
        print(f"Starting 10-fold CV ensemble (Strict Batch Training on {n_gpus} GPUs)...")

        # Calculate pos_counts
        pos_counts_np = np.sum(y_train_np, axis=0).astype(np.float32)
        pos_weights_np = (len(y_train_np) - pos_counts_np) / (pos_counts_np + 1e-5)

        folds_data = list(kf.split(x_train_np))

        # Train a single fold on a GPU device
        def train_single_fold(fold_idx, fold_gpu_idx, train_idx, val_idx):
            trial_device = torch.device(f'cuda:{fold_gpu_idx}')
            torch.cuda.set_device(trial_device)
            torch.cuda.empty_cache()

            try:
                # Move data tensors to target device
                x_tr_fold = torch.tensor(x_train_np[train_idx], dtype=torch.float32).to(trial_device)
                y_tr_fold = torch.tensor(y_train_np[train_idx], dtype=torch.float32).to(trial_device)
                x_test_tensor_local = torch.tensor(x_test_np, dtype=torch.float32).to(trial_device)
                fold_pos_weight = torch.tensor(pos_weights_np, dtype=torch.float32).to(trial_device)

                # Initialize model and optimizer
                model = self.create_model(hparams=best_params, input_dim=x_train_np.shape[1], num_tasks=num_tasks)

                if isinstance(model, nn.DataParallel):
                    model = model.module.to(trial_device)
                else:
                    model = model.to(trial_device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'],
                                              weight_decay=best_params['weight_decay'])
                criterion = nn.BCEWithLogitsLoss(pos_weight=fold_pos_weight)

                # Mixed precision setup
                scaler = torch.amp.GradScaler(device=trial_device.type)

                epoch_pbar = tqdm(
                    range(self.final_epochs),
                    desc=f"[GPU {fold_gpu_idx}] Fold {fold_idx + 1}/{n_folds}",
                    leave=True,
                    mininterval=5.0
                )

                for epoch in epoch_pbar:
                    model.train()
                    perm = torch.randperm(len(x_tr_fold), device=trial_device)

                    for i in range(0, len(x_tr_fold), self.batch_size):
                        idx = perm[i:i + self.batch_size]
                        x_batch = x_tr_fold[idx]
                        y_batch = y_tr_fold[idx]

                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast(device_type='cuda'):
                            task_pred, _ = model(x_batch)
                            loss = criterion(task_pred, y_batch)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                # Predict on test set for the current fold
                model.eval()
                all_preds = []
                with torch.no_grad():
                    for i in range(0, len(x_test_tensor_local), self.batch_size):
                        x_test_batch = x_test_tensor_local[i:i + self.batch_size]

                        with torch.amp.autocast(device_type='cuda'):
                            outputs = model(x_test_batch)
                            if isinstance(outputs, tuple):
                                task_pred_batch = outputs[0]
                            else:
                                task_pred_batch = outputs

                            all_preds.append(task_pred_batch.cpu())

                    task_pred_full = torch.cat(all_preds, dim=0)
                    prob = torch.sigmoid(task_pred_full).numpy()
                    fold_probs[fold_idx] = prob

            finally:
                # Memory cleanup and anti-OOM handling
                try:
                    if 'model' in locals():
                        model.to('cpu')
                    del model, optimizer, scaler, x_tr_fold, y_tr_fold, x_test_tensor_local, fold_pos_weight
                except NameError:
                    pass

                gc.collect()

                if 'trial_device' in locals():
                    torch.cuda.set_device(trial_device)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Batch scheduler for processing folds across available GPUs
        for i in range(0, n_folds, n_gpus):
            batch_folds = list(range(i, min(i + n_gpus, n_folds)))
            print(f"\n[Scheduler] Batch: Folds {[f + 1 for f in batch_folds]}")

            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = []
                for j, fold_idx in enumerate(batch_folds):
                    gpu_id = j % n_gpus
                    train_idx, val_idx = folds_data[fold_idx]

                    futures.append(
                        executor.submit(train_single_fold, fold_idx, gpu_id, train_idx, val_idx)
                    )
                # Wait for all folds in the current batch to finish
                for future in futures:
                    future.result()

            # Clean up GPU memory between batches
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        # Post-processing and ensemble evaluation
        gc.collect()
        torch.cuda.empty_cache()

        ensemble_prob = np.mean(fold_probs, axis=0)
        bin_pred = (ensemble_prob > 0.5).astype(int)

        task_names = self.categories if hasattr(self, 'categories') else \
            ['AB', 'ACE', 'ACP', 'AF', 'AMP', 'AOX', 'APP', 'AV', 'BBP', 'DPPIV', 'MRSA', 'NEURO', 'QS', 'TOX', 'TTCA',
             'AMAP']

        ensemble_mccs = []
        for j, name in enumerate(task_names[:num_tasks]):
            true = y_test_np[:, j]
            mcc = matthews_corrcoef(true, bin_pred[:, j]) if true.sum() > 0 else np.nan
            ensemble_mccs.append(mcc)

        overall_mcc = np.nanmean(ensemble_mccs)
        np.save(f"{outputdir}/run{self.current_run_idx + 1}_ensemble_probs.npy", ensemble_prob)

        print(f"\n10-Fold Ensemble (Hold-out Test) overall MCC: {overall_mcc:.4f}")
        return overall_mcc, ensemble_mccs, task_names[:num_tasks], ensemble_prob

    def calculate_val_mcc(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the average Matthews Correlation Coefficient (MCC) on the validation set for a single model.
        Args:
            probs: Predicted probability matrix, shape (n_samples, n_tasks).
            y_true: Ground truth label matrix, shape (n_samples, n_tasks).
        Returns:
            float: Mean MCC score across all valid tasks with positive samples.
        """

        # Convert prediction probabilities to binary predictions with default threshold 0.5
        bin_pred = (probs > 0.5).astype(int)
        mccs = []
        for j in range(y_true.shape[1]):
            # Calculate MCC only for tasks with positive samples to avoid zero-division issues
            if y_true[:, j].sum() > 0:
                mcc = matthews_corrcoef(y_true[:, j], bin_pred[:, j])
                mccs.append(mcc)
        return np.nanmean(mccs) if mccs else 0.0

    def train_model(self,
                     df_train: pd.DataFrame,
                     df_test: pd.DataFrame,
                     train_emb: np.ndarray,
                     test_emb: np.ndarray,
                     outputdir: str,
                     task_name: str = None,
                     hpo_strategy: str = "dual",
                     hpo_trials: int = 10,
                     max_epochs_per_trial: int = 100,
                     final_epochs: int = 200,
                     seed: int = 42,
                     n_replicates: int = 3,
                     **kwargs) -> dict:
        """
        Train PepMTL model using multi-replicate HPO and K-fold ensemble soft-voting.
        Args:
            df_train: Training set metadata and labels.
            df_test: Test set metadata and labels.
            train_emb: Precomputed sequence embeddings for training set.
            test_emb: Precomputed sequence embeddings for test set.
            outputdir: Directory path to save output files and logs.
            task_name: Target task name for single-task training mode.
                If None, multi-task mode is activated. Defaults to None.
            hpo_strategy: Strategy name for hyperparameter optimization. Defaults to "dual".
            hpo_trials: Number of trials per HPO optimization method. Defaults to 10.
            max_epochs_per_trial: Maximum training epochs per trial during HPO. Defaults to 100.
            final_epochs: Training epochs per fold in the final ensemble stage. Defaults to 200.
            seed: Base random seed for reproducibility. Defaults to 42.
            n_replicates: Number of independent experimental replicates. Defaults to 3.
        Returns:
            dict: Summary dictionary containing hold-out test metrics, per-task mean and SEM,
                ensemble status, replicate counts, and generated plot paths.
        """
        start_time = time.time()
        print(f"\nStarting PepMTL Training...")

        self.gpu_lock = threading.Lock()
        self.gpu_availability = {i: True for i in range(torch.cuda.device_count())}

        num_tasks = 1 if task_name is not None else 16
        batch_size = 1024

        # Detect usable GPU instances in system
        n_gpus = torch.cuda.device_count()
        print(f"Detected {n_gpus} usable GPU(s).")

        self.final_epochs = final_epochs
        self.batch_size = batch_size
        self.pos_weight_tensor = None

        if task_name is not None:
            y_cols = [f'Y_{task_name}']
            num_tasks = 1
            print(f"Training in single-task mode for: {task_name}")
        else:
            y_cols = [f'Y_{cat}' for cat in self.categories]
            num_tasks = len(y_cols)
            #print(f"Training in multi-task mode with {num_tasks} tasks")

        replicate_overall_mccs = []
        replicate_per_task_mccs = []
        replicate_per_task_thresholds = []

        for rep in range(n_replicates):
            print(f"\nReplicate {rep + 1}/{n_replicates} (seed={seed + rep * 100})")
            current_seed = seed + rep * 100
            self.current_run_idx = rep

            x_train_np = train_emb
            y_train_np = df_train[y_cols].values
            x_test_np = test_emb
            y_test_np = df_test[y_cols].values

            # Prepare validation dataset split for HPO phase
            x_tr, x_val, y_tr, y_val = train_test_split(x_train_np, y_train_np, test_size=0.1,
                                                        random_state=current_seed)
            y_val_np = y_val

            pos_counts_np = np.sum(y_tr, axis=0).astype(np.float32)
            neg_counts_np = len(y_tr) - pos_counts_np
            pos_ratio_np = neg_counts_np / np.clip(pos_counts_np, 1.0, None)
            pos_weights_np = np.clip(pos_ratio_np, 1.0, 140.0)

            # HPO with Random Search and Bayesian Optimization
            def objective(trial):
                trial_gpu_idx = None
                while trial_gpu_idx is None:
                    with self.gpu_lock:
                        for gpu_id, is_free in self.gpu_availability.items():
                            if is_free:
                                trial_gpu_idx = gpu_id
                                self.gpu_availability[gpu_id] = False
                                break
                    if trial_gpu_idx is None:
                        time.sleep(0.5)

                trial_device = torch.device(f'cuda:{trial_gpu_idx}')
                torch.cuda.set_device(trial_device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                hparams = {
                    'lstm_num_layers': trial.suggest_int('lstm_num_layers', 2, 3),
                    'lstm_hidden': trial.suggest_categorical('lstm_hidden', [512, 768, 1024]),
                    'cnn_out': trial.suggest_categorical('cnn_out', [384, 512, 768]),
                    'cnn_kernel': trial.suggest_int('cnn_kernel', 3, 5),
                    'head_hidden': trial.suggest_categorical('head_hidden', [384, 512, 768]),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                    'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 8e-4, log=True),
                }

                try:
                    x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32).to(trial_device)
                    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(trial_device)
                    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(trial_device)
                    trial_pos_weight = torch.tensor(pos_weights_np, dtype=torch.float32, device=trial_device)

                    raw_model = self.create_model(hparams, x_train_np.shape[1], num_tasks)
                    if isinstance(raw_model, nn.DataParallel):
                        model = raw_model.module.to(trial_device)
                    else:
                        model = raw_model.to(trial_device)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'],
                                                  weight_decay=hparams['weight_decay'])
                    criterion = nn.BCEWithLogitsLoss(pos_weight=trial_pos_weight)
                    scaler = torch.amp.GradScaler(device=trial_device.type)

                    best_val_mcc = -1.0
                    patience_cnt = 0
                    patience = 10
                    min_delta = 0.001

                    gpu_bar_position = trial_gpu_idx + 1
                    epoch_pbar = tqdm(
                        range(max_epochs_per_trial),
                        desc=f" [GPU {trial_gpu_idx}] Trial {trial.number}",
                        position=gpu_bar_position,
                        leave=False,
                        mininterval=1.0
                    )

                    for epoch in epoch_pbar:
                        model.train()
                        perm = torch.randperm(len(x_tr_tensor), device=trial_device)
                        for i in range(0, len(x_tr_tensor), batch_size):
                            idx = perm[i:i + batch_size]
                            x_batch = x_tr_tensor[idx]
                            y_batch = y_tr_tensor[idx]

                            optimizer.zero_grad(set_to_none=True)
                            with torch.amp.autocast(device_type='cuda'):
                                task_pred, _ = model(x_batch)
                                loss = criterion(task_pred, y_batch)

                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()

                        val_mcc = self.evaluate_model(
                            model=model, x_data=x_val_tensor, y_data=y_val_np,
                            batch_size=batch_size, device=trial_device, num_tasks=num_tasks
                        )

                        if val_mcc > best_val_mcc + min_delta:
                            best_val_mcc = val_mcc
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                            if patience_cnt >= patience:
                                break

                    del x_tr_tensor, y_tr_tensor, x_val_tensor, model, optimizer, scaler
                    torch.cuda.empty_cache()

                    return best_val_mcc

                finally:
                    with self.gpu_lock:
                        self.gpu_availability[trial_gpu_idx] = True

            # Execute Hyperparameter Optimization (HPO)
            hpo_results = {}

            print(f"\nRunning Random Search HPO (Parallel on {n_gpus} GPUs):")
            study_random = optuna.create_study(direction="maximize", sampler=RandomSampler(current_seed))
            study_random.optimize(objective, n_trials=hpo_trials, n_jobs=1)
            joblib.dump(study_random, os.path.join(outputdir, f"optuna_random_rep_{rep}.pkl"))

            hpo_results['Random'] = {
                'best_value': study_random.best_value,
                'best_params': study_random.best_params,
                'n_trials': len(study_random.trials)
            }

            print("\nRunning Bayesian Optimization HPO:")
            study_tpe = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(
                    seed=current_seed,
                    multivariate=True,
                    group=True
                )
            )

            study_tpe.optimize(objective, n_trials=hpo_trials, n_jobs=1)
            joblib.dump(study_tpe, os.path.join(outputdir, f"optuna_study_rep_{rep}.pkl"))

            hpo_results['Bayesian'] = {
                'best_value': study_tpe.best_value,
                'best_params': study_tpe.best_params,
                'n_trials': len(study_tpe.trials)
            }

            # Compare performance between Random Search and Bayesian Optimization
            winner = 'Random' if hpo_results['Random']['best_value'] > hpo_results['Bayesian']['best_value'] else 'Bayesian'
            best_params = study_tpe.best_params if winner == 'Bayesian' else study_random.best_params

            print("\n" + "=" * 80)
            title = f"HPO Method Comparison (each method runs {hpo_trials} trials)"
            print(title.center(80))
            print("=" * 80)

            col_widths = [18, 20, 10, 22]

            headers = [
                "Method".center(col_widths[0]),
                "Best MCC".center(col_widths[1]),
                "Trials".center(col_widths[2]),
                "Recommendation".center(col_widths[3])
            ]
            print("".join(headers))
            print("-" * 80)

            for method in ['Random', 'Bayesian']:
                res = hpo_results[method]
                recommend = " Recommended" if method == winner else ""

                row = [
                    method.center(col_widths[0]),
                    f"{res['best_value']:.4f}".center(col_widths[1]),
                    str(res['n_trials']).center(col_widths[2]),
                    recommend.center(col_widths[3])
                ]
                line = "".join(row)

                if method == winner:
                    print(f"\033[1m{line}\033[0m")
                else:
                    print(line)

            print("-" * 80)

            diff = abs(hpo_results['Random']['best_value'] - hpo_results['Bayesian']['best_value'])
            conclusion = f"Conclusion: {winner} performs better (MCC higher by {diff:.4f})"
            print(conclusion.center(80))

            print(f"\nUsing the best parameters from {winner} for final training:")
            print(str(best_params))

            self.device = torch.device('cuda:0')
            self.pos_weight_tensor = torch.tensor(pos_weights_np, dtype=torch.float32, device=self.device)

            # Final Model Ensemble Training ===
            ensemble_mcc, per_task_mccs, task_names, ensemble_prob = self.ensemble(
                best_params, x_train_np, y_train_np, x_test_np, y_test_np,
                outputdir, current_seed, num_tasks
            )
            replicate_overall_mccs.append(ensemble_mcc)
            replicate_per_task_mccs.append(per_task_mccs)  # 收集 per-task

        # Final Performance Aggregation
        replicate_per_task_mccs = np.array(replicate_per_task_mccs)
        per_task_mean = np.nanmean(replicate_per_task_mccs, axis=0)
        per_task_sem = np.nanstd(replicate_per_task_mccs, axis=0, ddof=1) / np.sqrt(n_replicates)

        overall_mean = np.nanmean(per_task_mean)
        overall_sem = np.std(replicate_overall_mccs, ddof=1) / np.sqrt(n_replicates)

        task_names_for_dict = task_names
        per_task_mean_dict = dict(zip(task_names_for_dict, per_task_mean.tolist()))
        per_task_sem_dict = dict(zip(task_names_for_dict, per_task_sem.tolist()))

        end_time = time.time()
        total_seconds = end_time - start_time
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60

        # Print training summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print(f"Total execution time: {total_hours:.2f} hours "
              f"({total_minutes:.1f} minutes / {total_seconds:.0f} seconds)")
        print("=" * 80)

        # Write summary log file for scientific reporting
        time_log_path = os.path.join(outputdir, "summary_time.log")
        with open(time_log_path, "w", encoding="utf-8") as f:
            f.write("E2EPepMTL Training Summary\n")
            f.write("-" * 40 + "\n")
            f.write(f"Start time          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"End time            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"Total duration      : {total_hours:.2f} hours ({total_seconds:.0f} seconds)\n")
            f.write(f"Hyperparameter trials: {hpo_trials}\n")
            f.write(f"Runs per trial      : {n_replicates}\n")
            f.write(
                f"Training mode       : {'Single-task' if task_name else f'Multi-task ({len(self.categories)} bioactivities)'}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Output directory    : {outputdir}\n")

        # Display hold-out test results
        print("\n" + "=" * 40)
        print("Hold-out Test Mean (MCC ± SEM)".center(40))
        print("=" * 40)
        print(f"{'Task':<8} {'Mean MCC':<10} ± SEM")
        print("-" * 40)
        for name, m, s in zip(task_names_for_dict, per_task_mean, per_task_sem):
            print(f"{name:<8} {m:8.4f}  ± {s:.4f}")
        print("-" * 40)
        print(f"{'Overall':<8} {overall_mean:8.4f}  ± {overall_sem:.4f}")

        # Plot performance metrics
        plot_results = self.plot_mtl_performance(
            best_task_mccs=per_task_mean.tolist(),
            best_mcc_final=overall_mean,
            best_epoch="10-fold Ensemble",
            outputdir=outputdir,
            arch_name="PepMTL",
            hpo_name="Dual HPO + 10-fold Soft Voting",
            n_runs=n_replicates,
            sem_test_mcc=overall_sem,
            task_sem=per_task_sem.tolist(),
            task_name=task_name,
            is_holdout=True
        )

        # Construct final pipeline results
        pipeline_results = {
            'mean_mcc_test': float(overall_mean),
            'sem_mcc_test': float(overall_sem),
            'per_task_mean': per_task_mean_dict,
            'per_task_sem': per_task_sem_dict,
            'ensemble_used': True,
            'replicates': n_replicates,
            'plot_files': plot_results
        }

        # Save raw replicate MCC values to CSV
        raw_mcc_df = pd.DataFrame(
            replicate_per_task_mccs,
            columns=task_names_for_dict,
            index=[f'Seed_{seed + r * 100}' for r in range(n_replicates)]
        )
        raw_mcc_df['Overall_Average'] = replicate_overall_mccs
        raw_data_path = os.path.join(outputdir, "raw_replicate_mccs.csv")
        raw_mcc_df.to_csv(raw_data_path)

        print(f"\nRaw MCC values (Tasks & Overall) saved to: {raw_data_path}")

        return pipeline_results

    def train_stl(self,
                  df_train: pd.DataFrame,
                  df_test: pd.DataFrame,
                  train_emb: np.ndarray,
                  test_emb: np.ndarray,
                  outputdir: str,
                  task_name: str,
                  hpo_trials: int = 10,
                  max_epochs_per_trial: int = 100,
                  final_epochs: int = 200,
                  seed: int = 42,
                  n_replicates: int = 3,
                  backbone_type: str = "cnn",  # hybrid, bilstm, cnn
                  **kwargs) -> dict:
        """
        Train Single-Task Learning (STL) models for a specified bioactivity task.
        Args:
            df_train: Training metadata containing target labels.
            df_test: Test metadata containing target labels.
            train_emb: Precomputed sequence embeddings for training set.
            test_emb: Precomputed sequence embeddings for test set.
            outputdir: Directory path to save output summary logs and results.
            task_name: Specific bioactivity task name for single-task training.
            hpo_trials: Number of trials per HPO optimization method. Defaults to 10.
            max_epochs_per_trial: Maximum training epochs per trial during HPO. Defaults to 100.
            final_epochs: Epoch count for final model training using best parameters. Defaults to 200.
            seed: Base random seed for experimental reproducibility. Defaults to 42.
            n_replicates: Number of independent training replicates. Defaults to 3.
            backbone_type: Neural backbone architecture type ('hybrid', 'bilstm', or 'cnn'). Defaults to "cnn".
        Returns:
            dict: Summary dictionary containing evaluation metrics, per-replicate MCCs, execution time, and HPO settings.
        Raises:
            ValueError: If the required target column `Y_{task_name}` is missing from `df_train`.
        """

        start_time = time.time()
        print(f"\n=== Starting STL {backbone_type} training for task: {task_name} ===")

        y_col = f'Y_{task_name}'
        if y_col not in df_train.columns:
            raise ValueError(f"Column {y_col} not found in df_train")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 1024

        replicate_mccs = []

        for rep in range(n_replicates):
            print(f"\nReplicate {rep + 1}/{n_replicates} (seed = {seed + rep * 100})")
            current_seed = seed + rep * 100

            # Data preparation and class weight computation
            y_train_np = df_train[y_col].values.astype(np.float32)  # shape: (n,)
            y_test_np = df_test[y_col].values.astype(np.float32)

            # pos_weight (scalar)
            pos_count = np.sum(y_train_np)
            neg_count = len(y_train_np) - pos_count
            pos_weight = neg_count / max(1.0, pos_count)
            pos_weight = min(pos_weight, 140.0)
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

            print(f"  Pos: {int(pos_count):,} | Neg: {int(neg_count):,} | pos_weight: {pos_weight:.2f}")

            # Train and validation set split for HPO
            x_tr, x_val, y_tr, y_val = train_test_split(
                train_emb, y_train_np, test_size=0.1, random_state=current_seed
            )

            x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32).to(device)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
            x_test_tensor = torch.tensor(test_emb, dtype=torch.float32).to(device)

            # HPO objective
            def objective(trial):
                torch.cuda.empty_cache()

                hparams = {
                    'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                    'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 8e-4, log=True),
                    'head_hidden': trial.suggest_categorical('head_hidden', [384, 512, 768]),
                }

                if backbone_type in ["hybrid", "bilstm"]:
                    hparams['lstm_num_layers'] = trial.suggest_int('lstm_num_layers', 2, 3)
                    hparams['lstm_hidden'] = trial.suggest_categorical('lstm_hidden', [512, 768, 1024])

                if backbone_type in ["hybrid", "cnn"]:
                    hparams['cnn_out'] = trial.suggest_categorical('cnn_out', [384, 512, 768])
                    hparams['cnn_kernel'] = trial.suggest_int('cnn_kernel', 3, 5)

                # Initialize single-task model architecture
                model = self.create_model_stl(hparams, train_emb.shape[1], backbone_type=backbone_type)
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'],
                                              weight_decay=hparams['weight_decay'])
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                scaler = torch.amp.GradScaler('cuda')

                best_val_mcc = -1.0
                patience_cnt = 0
                patience = 12
                min_delta = 0.001

                for epoch in tqdm(range(max_epochs_per_trial),
                                  desc=f"Trial {trial.number}",
                                  leave=False, mininterval=2):
                    model.train()
                    perm = torch.randperm(len(x_tr_tensor))
                    for i in range(0, len(x_tr_tensor), batch_size):
                        idx = perm[i:i + batch_size]
                        x_b = x_tr_tensor[idx]
                        y_b = y_tr_tensor[idx]

                        optimizer.zero_grad()
                        # with torch.amp.autocast('cuda'):
                        #     pred, _ = model(x_b)
                        #     pred = pred.squeeze(-1)
                        #     loss = criterion(pred, y_b)

                        with torch.amp.autocast('cuda'):
                            output = model(x_b)
                            if isinstance(output, tuple):
                                pred = output[0].squeeze(-1)  # 如果是 (pred, None)
                            else:
                                pred = output.squeeze(-1)  # 如果只回一個 tensor
                            loss = criterion(pred, y_b)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    # Evaluate trial model performance on validation set
                    model.eval()
                    with torch.no_grad():
                        val_pred = []
                        for i in range(0, len(x_val_tensor), batch_size):
                            x_vb = x_val_tensor[i:i + batch_size]
                            # out, _ = model(x_vb)
                            # out = out.squeeze(-1)
                            # val_pred.append(out.cpu().numpy())
                            out = model(x_vb)
                            if isinstance(out, tuple):
                                out = out[0]
                            out = out.squeeze(-1)
                            val_pred.append(out.cpu().numpy())
                        val_pred = np.concatenate(val_pred)
                        val_pred_bin = (val_pred > 0).astype(int)
                        val_mcc = matthews_corrcoef(y_val, val_pred_bin)

                    if val_mcc > best_val_mcc + min_delta:
                        best_val_mcc = val_mcc
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

                return best_val_mcc

            # Execute dual HPO strategies
            print("  Running Random Search...")
            study_random = optuna.create_study(direction="maximize", sampler=RandomSampler(current_seed))
            study_random.optimize(objective, n_trials=hpo_trials)

            print("  Running Bayesian...")
            study_tpe = optuna.create_study(direction="maximize", sampler=TPESampler(seed=current_seed))
            study_tpe.optimize(objective, n_trials=hpo_trials)

            # Select the winning parameter set
            best_value = max(study_random.best_value, study_tpe.best_value)
            if study_random.best_value > study_tpe.best_value:
                best_params = study_random.best_params
                winner = "Random Search"
            else:
                best_params = study_tpe.best_params
                winner = "Bayesian"

            print(f"Winner: {winner} (best val MCC = {best_value:.4f})")
            print(f"{best_params}")

            # Train final model using optimal hyperparameters
            print("  Final training with best params...")
            model = self.create_model_stl(best_params, train_emb.shape[1], backbone_type=backbone_type)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'],
                                          weight_decay=best_params['weight_decay'])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            scaler = torch.amp.GradScaler('cuda')

            for epoch in range(final_epochs):
                model.train()
                perm = torch.randperm(len(x_tr_tensor))
                for i in range(0, len(x_tr_tensor), batch_size):
                    idx = perm[i:i + batch_size]
                    x_b = x_tr_tensor[idx]
                    y_b = y_tr_tensor[idx]

                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        pred, _ = model(x_b)
                        pred = pred.squeeze(-1)
                        loss = criterion(pred, y_b)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

            # Evaluate model performance on test set
            model.eval()
            with torch.no_grad():
                test_pred = []
                for i in range(0, len(x_test_tensor), batch_size):
                    x_tb = x_test_tensor[i:i + batch_size]
                    # out, _ = model(x_tb)
                    # task_pred = out.squeeze(-1)
                    # test_pred.append(out.cpu().numpy())
                    out = model(x_tb)
                    if isinstance(out, tuple):
                        out = out[0]
                    task_pred = out.squeeze(-1)
                    test_pred.append(task_pred.cpu().numpy())
                test_pred = np.concatenate(test_pred)
                test_pred_bin = (test_pred > 0).astype(int)
                test_mcc = matthews_corrcoef(y_test_np, test_pred_bin)

            replicate_mccs.append(test_mcc)
            print(f"  Replicate {rep + 1} Test MCC: {test_mcc:.4f}")

        # Aggregate performance across replicates
        mean_mcc = np.mean(replicate_mccs)
        sem_mcc = np.std(replicate_mccs, ddof=1) / np.sqrt(n_replicates) if n_replicates > 1 else 0.0

        end_time = time.time()
        total_seconds = end_time - start_time
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print(f"Total execution time: {total_hours:.2f} hours "
              f"({total_minutes:.1f} minutes / {total_seconds:.0f} seconds)")

        # Save summary execution log file
        time_log_path = os.path.join(outputdir, "summary_time.log")
        with open(time_log_path, "w", encoding="utf-8") as f:
            f.write("E2EPepMTL Training Summary\n")
            f.write("-" * 40 + "\n")
            f.write(f"Start time          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"End time            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"Total duration      : {total_hours:.2f} hours ({total_seconds:.0f} seconds)\n")
            f.write(f"Hyperparameter trials: {hpo_trials}\n")
            f.write(f"Runs per trial      : {n_replicates}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Output directory    : {outputdir}\n")

        print(f"\nSTL Final Result for {task_name}:")
        print(f"Mean MCC ± SEM ({n_replicates} replicates): {mean_mcc:.4f} ± {sem_mcc:.4f}")

        results = {
            'task': task_name,
            'backbone': backbone_type,
            'mean_mcc': float(mean_mcc),
            'sem_mcc': float(sem_mcc),
            'replicate_mccs': replicate_mccs,
            'n_replicates': n_replicates,
            'total_time_hours': total_hours,
            'hpo_trials': hpo_trials
        }

        return results

    def train_hybrid(self,
                     df_train: pd.DataFrame,
                     df_test: pd.DataFrame,
                     train_emb: np.ndarray,
                     test_emb: np.ndarray,
                     outputdir: str,
                     task_name: str = None,
                     hpo_strategy: str = "dual",
                     hpo_trials: int = 10,
                     max_epochs_per_trial: int = 100,
                     final_epochs: int = 200,
                     seed: int = 42,
                     n_replicates: int = 3,
                     **kwargs
                     ) -> dict:
        """
        Train a PepMTL + Support Vector Machine + K-Nearest Neighbors Hybrid Ensemble model.

        Args:
            df_train: Metadata containing training target labels.
            df_test: Metadata containing testing target labels.
            train_emb: Precomputed feature embeddings for training set.
            test_emb: Precomputed feature embeddings for testing set.
            outputdir: Directory path to save output models, probabilities, logs, and plots.
            task_name: Name of a specific bioactivity task if training single-task,
                otherwise None for multi-task. Defaults to None.
            hpo_strategy: Strategy for hyperparameter tuning. Defaults to "dual".
            hpo_trials: Number of optimization trials per HPO sampler. Defaults to 10.
            max_epochs_per_trial: Maximum training epochs per trial during HPO. Defaults to 100.
            final_epochs: Epoch limit for training validation baseline models. Defaults to 200.
            seed: Base seed for random number generation and reproducibility. Defaults to 42.
            n_replicates: Number of independent experimental runs. Defaults to 3.
        Returns:
            dict: A dictionary containing overall and per-task mean MCCs, SEMs, ensemble flags,
                and generated plot file paths.
        """
        start_time = time.time()
        print(f"\nStarting PepMTL + SVM + KNN Hybrid Ensemble...")

        num_tasks = 1 if task_name is not None else 16
        batch_size = 512
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.final_epochs = final_epochs
        self.batch_size = batch_size
        self.pos_weight_tensor = None

        # Determine task columns and names
        if task_name is not None:
            y_cols = [f'Y_{task_name}']
            task_names = [task_name]
        else:
            y_cols = [f'Y_{cat}' for cat in self.categories]
            task_names = self.categories

        replicate_overall_mccs = []
        replicate_per_task_mccs = []

        for rep in range(n_replicates):
            print(f"\nReplicate {rep + 1}/{n_replicates} (seed={seed + rep * 100})")
            current_seed = seed + rep * 100
            self.current_run_idx = rep

            x_train_np = train_emb
            y_train_np = df_train[y_cols].values
            x_test_np = test_emb
            y_test_np = df_test[y_cols].values

            # Calculate positive sample weight
            x_tr, x_val, y_tr, y_val = train_test_split(x_train_np, y_train_np, test_size=0.1,
                                                        random_state=current_seed)
            x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32).to(device)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
            y_val_np = y_val

            pos_counts_np = np.sum(y_tr, axis=0).astype(np.float32)
            neg_counts_np = len(y_tr) - pos_counts_np
            pos_ratio_np = neg_counts_np / np.clip(pos_counts_np, 1.0, None)
            pos_weights_np = np.clip(pos_ratio_np, 1.0, 400.0)
            self.pos_weight_tensor = torch.tensor(pos_weights_np, dtype=torch.float32, device=device)

            # Hyperparameter Optimization (HPO)
            def objective(trial):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                hparams = {
                    'lstm_num_layers': trial.suggest_int('lstm_num_layers', 2, 3),
                    'lstm_hidden': trial.suggest_categorical('lstm_hidden', [512, 768, 1024]),
                    'cnn_out': trial.suggest_categorical('cnn_out', [384, 512, 768]),
                    'cnn_kernel': trial.suggest_int('cnn_kernel', 3, 5),
                    'head_hidden': trial.suggest_categorical('head_hidden', [384, 512, 768]),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                    'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 8e-4, log=True),
                }

                model = self.create_model(
                                hparams,
                                x_train_np.shape[1],
                                num_tasks,
                                task_groups=task_groups
                                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'],
                                              weight_decay=hparams['weight_decay'])
                criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
                scaler = torch.amp.GradScaler('cuda')

                best_val_mcc = -1.0
                patience_cnt = 0
                patience = 10
                min_delta = 0.001

                for epoch in tqdm(range(max_epochs_per_trial),
                                  desc=f"Trial {trial.number} Epochs",
                                  unit="epoch",
                                  leave=False,
                                  mininterval=2):
                    model.train()
                    perm = torch.randperm(len(x_tr_tensor))
                    for i in range(0, len(x_tr_tensor), batch_size):
                        idx = perm[i:i + batch_size]
                        x_batch = x_tr_tensor[idx]
                        y_batch = y_tr_tensor[idx]

                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda'):
                            task_pred, _ = model(x_batch)
                            loss = criterion(task_pred, y_batch)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    val_mcc = self.evaluate_model(
                        model=model,
                        x_data=x_val_tensor,
                        y_data=y_val_np,
                        batch_size=batch_size,
                        device=device,
                        num_tasks=num_tasks
                    )

                    if val_mcc > best_val_mcc + min_delta:
                        best_val_mcc = val_mcc
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

                return best_val_mcc

            # Execute Dual HPO
            hpo_results = {}

            print("\nRunning Random Search HPO:")
            study_random = optuna.create_study(direction="maximize", sampler=RandomSampler(current_seed))
            with tqdm(total=hpo_trials, desc="Random HPO", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix(best=f"{study.best_value:.4f}")

                study_random.optimize(objective, n_trials=hpo_trials, callbacks=[callback])

            hpo_results['Random'] = {
                'best_value': study_random.best_value,
                'best_params': study_random.best_params,
                'n_trials': len(study_random.trials)
            }

            print("\nRunning Bayesian Optimization HPO:")
            study_tpe = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(
                    seed=current_seed,
                    multivariate=True,
                    group=True
                )
            )
            with tqdm(total=hpo_trials, desc="Bayesian HPO", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix(best=f"{study.best_value:.4f}")

                study_tpe.optimize(objective, n_trials=hpo_trials, callbacks=[callback])

            hpo_results['Bayesian'] = {
                'best_value': study_tpe.best_value,
                'best_params': study_tpe.best_params,
                'n_trials': len(study_tpe.trials)
            }

            # Compare and select winning HPO strategy
            winner = 'Random' if hpo_results['Random']['best_value'] > hpo_results['Bayesian'][
                'best_value'] else 'Bayesian'
            best_params = study_tpe.best_params if winner == 'Bayesian' else study_random.best_params

            print("\n" + "=" * 80)
            title = f"HPO Method Comparison (each method runs {hpo_trials} trials)"
            print(title.center(80))
            print("=" * 80)

            col_widths = [18, 20, 10, 22]

            headers = [
                "Method".center(col_widths[0]),
                "Best MCC".center(col_widths[1]),
                "Trials".center(col_widths[2]),
                "Recommendation".center(col_widths[3])
            ]
            print("".join(headers))
            print("-" * 80)

            for method in ['Random', 'Bayesian']:
                res = hpo_results[method]
                recommend = " Recommended" if method == winner else ""

                row = [
                    method.center(col_widths[0]),
                    f"{res['best_value']:.4f}".center(col_widths[1]),
                    str(res['n_trials']).center(col_widths[2]),
                    recommend.center(col_widths[3])
                ]
                line = "".join(row)

                if method == winner:
                    print(f"\033[1m{line}\033[0m")
                else:
                    print(line)

            print("-" * 80)

            diff = abs(hpo_results['Random']['best_value'] - hpo_results['Bayesian']['best_value'])
            conclusion = f"Conclusion: {winner} performs better (MCC higher by {diff:.4f})"
            print(conclusion.center(80))

            print(f"\nUsing the best parameters from {winner} for final training:")
            print(str(best_params))

            # PepMTL 10-fold Cross-Validation Ensemble
            ensemble_mcc_dl, per_task_mccs_dl, task_names, dl_prob = self.ensemble3(
                best_params, x_train_np, y_train_np, x_test_np, y_test_np,
                outputdir, current_seed, num_tasks
            )

            # Evaluate Models on Validation Set for Dynamic Weighting
            print("\n" + "=" * 80)
            print("→ Computing validation set predictions for dynamic weight assignment...")
            print("=" * 80)

            # PepMTL (DL) Validation Prediction
            print("[1/4] Computing PepMTL (DL) validation predictions...")

            val_model = self.create_model3(best_params, x_train_np.shape[1], num_tasks, task_groups=task_groups)
            if isinstance(val_model, PepMTL):
                val_model.set_task_groups(task_groups)

            val_model = self.create_model(best_params, x_train_np.shape[1], num_tasks)
            val_model.train()
            optimizer = torch.optim.AdamW(val_model.parameters(),
                                          lr=best_params['lr'],
                                          weight_decay=best_params['weight_decay'])
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
            scaler = torch.amp.GradScaler('cuda')

            print("Training val_model on full training set...")
            best_val_mcc_val = -1.0
            patience_cnt = 0
            patience = 10
            for epoch in tqdm(range(80), desc="Val Model Training", unit="epoch", leave=False):
                val_model.train()
                perm = torch.randperm(len(x_train_np))
                x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(self.device)

                for i in range(0, len(x_train_np), batch_size):
                    idx = perm[i:i + batch_size]
                    x_batch = x_train_tensor[idx]
                    y_batch = y_train_tensor[idx]

                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        task_pred, _ = val_model(x_batch)
                        loss = criterion(task_pred, y_batch)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(val_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                val_mcc_val = self.evaluate_model(
                    model=val_model,
                    x_data=x_val_tensor,
                    y_data=y_val_np,
                    batch_size=batch_size,
                    device=device,
                    num_tasks=num_tasks
                )

                if val_mcc_val > best_val_mcc_val + 0.001:
                    best_val_mcc_val = val_mcc_val
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

            val_model.eval()
            with torch.no_grad():
                dl_val_pred, _ = val_model(x_val_tensor)
                dl_val_prob = torch.sigmoid(dl_val_pred).cpu().numpy()

            tqdm(total=1, desc="PepMTL (DL) Validation", unit="step", leave=True).update(1)
            print("✓ PepMTL (DL) validation predictions completed.")

            # Feature Standardization
            print("[2/4] Standardizing feature embeddings...")
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_np)
            x_val_scaled = scaler.transform(x_val)
            x_test_scaled = scaler.transform(x_test_np)
            tqdm(total=1, desc="Feature Scaling", unit="step", leave=True).update(1)
            print("✓ Feature standardization completed.")

            #  SVM Validation Prediction
            print("[3/4] Computing SVM validation predictions...")
            svm_val_prob = np.zeros_like(dl_val_prob)
            for j in tqdm(range(num_tasks), desc="SVM Validation", unit="task", leave=True):
               y_tr = y_train_np[:, j]
               svm = SVC(probability=True, kernel='rbf', C=1.0,
                         cache_size=2000, random_state=current_seed)
               svm.fit(x_train_scaled, y_tr)
               svm_val_prob[:, j] = svm.predict_proba(x_val_scaled)[:, 1]
            print("✓ SVM validation predictions completed.")

            # KNN Validation Prediction
            print("[4/4] Computing KNN validation predictions...")
            knn_val_prob = np.zeros_like(dl_val_prob)
            for j in tqdm(range(num_tasks), desc="KNN Validation", unit="task", leave=True):
               y_tr = y_train_np[:, j]
               knn = KNeighborsClassifier(n_neighbors=5)
               knn.fit(x_train_scaled, y_tr)
               knn_val_prob[:, j] = knn.predict_proba(x_val_scaled)[:, 1]
            print("✓ KNN validation predictions completed.")

            # Evaluate validation performance across models
            print("\n" + "-" * 80)
            print("Model MCC scores on validation set (Higher -> Larger Ensemble Weight):")
            val_dl_mcc = self.calculate_val_mcc(dl_val_prob, y_val_np)
            val_dl_mcc = max(val_dl_mcc, 0.0)
            val_svm_mcc = self.calculate_val_mcc(svm_val_prob, y_val_np)
            val_knn_mcc = self.calculate_val_mcc(knn_val_prob, y_val_np)

            print(f"  PepMTL(DL) : {val_dl_mcc:.4f}")
            print(f"  SVM        : {val_svm_mcc:.4f}")
            print(f"  KNN        : {val_knn_mcc:.4f}")
            print("-" * 80)

            # Calculate dynamic ensemble weights
            total = val_dl_mcc + val_svm_mcc + val_knn_mcc + 1e-8
            w_dl = val_dl_mcc / total
            w_svm = val_svm_mcc / total
            w_knn = val_knn_mcc / total

            print(f"\nFinal Ensemble Weight Allocation:")
            print(f"  PepMTL(DL) Weight: {w_dl:.4f}  ({w_dl * 100:.1f}%)")
            print(f"  SVM        Weight: {w_svm:.4f}  ({w_svm * 100:.1f}%)")
            print(f"  KNN        Weight: {w_knn:.4f}  ({w_knn * 100:.1f}%)")
            print("=" * 70)

            # Final Inference on Test Set
            print("\nTraining SVM and KNN for final test set prediction...")
            print("→ Train SVM (test set)...")
            svm_prob = np.zeros_like(dl_prob)
            for j in tqdm(range(num_tasks), desc="SVM 測試集", unit="task"):
               y_tr = y_train_np[:, j]
               svm = SVC(probability=True, kernel='rbf', C=1.0,
                         class_weight='balanced', cache_size=2000, random_state=current_seed)
               svm.fit(x_train_scaled, y_tr)
               svm_prob[:, j] = svm.predict_proba(x_test_scaled)[:, 1]
            print("✓ SVM test set predictions completed.")

            print("→ Train KNN (test set)...")
            knn_prob = np.zeros_like(dl_prob)
            for j in tqdm(range(num_tasks), desc="KNN test set", unit="task"):
               y_tr = y_train_np[:, j]
               knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
               knn.fit(x_train_scaled, y_tr)
               knn_prob[:, j] = knn.predict_proba(x_test_scaled)[:, 1]
            print("✓ KNN test set predictions completed.")

            # Dynamic Soft Voting Fusion
            print("\nPerforming dynamic weighted probability fusion...")
            final_prob = (w_dl * dl_prob) + (w_svm * svm_prob) + (w_knn * knn_prob)
            bin_pred = (final_prob > 0.5).astype(int)

            # Compute test set MCC metrics
            ensemble_mccs = []
            for j in range(num_tasks):
                mcc = matthews_corrcoef(y_test_np[:, j], bin_pred[:, j]) if y_test_np[:, j].sum() > 0 else np.nan
                ensemble_mccs.append(mcc)

            ensemble_mcc = np.nanmean(ensemble_mccs)
            print(f"  Hybrid Dynamic Weighted Ensemble Overall MCC: {ensemble_mcc:.4f}")

            eplicate_overall_mccs.append(ensemble_mcc)
            replicate_per_task_mccs.append(ensemble_mccs)

        # Aggregate Replicate Performance
        replicate_per_task_mccs = np.array(replicate_per_task_mccs)
        per_task_mean = np.nanmean(replicate_per_task_mccs, axis=0)
        per_task_sem = np.nanstd(replicate_per_task_mccs, axis=0, ddof=1) / np.sqrt(n_replicates)

        overall_mean = np.nanmean(per_task_mean)
        overall_sem = np.std(replicate_overall_mccs, ddof=1) / np.sqrt(n_replicates)

        task_names_for_dict = task_names
        per_task_mcc_dict = dict(zip(task_names_for_dict, per_task_mean.tolist()))
        per_task_mean_dict = dict(zip(task_names_for_dict, per_task_mean.tolist()))
        per_task_sem_dict = dict(zip(task_names_for_dict, per_task_sem.tolist()))

        end_time = time.time()
        total_seconds = end_time - start_time
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60

        # Summary
        time_log_path = os.path.join(outputdir, "summary_time.log")
        with open(time_log_path, "w", encoding="utf-8") as f:
            f.write("E2EPepMTL Training Summary\n")
            f.write("-" * 40 + "\n")
            f.write(f"Start time          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"End time            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"Total duration      : {total_hours:.2f} hours ({total_seconds:.0f} seconds)\n")
            f.write(f"Hyperparameter trials: {hpo_trials}\n")
            f.write(f"Runs per trial      : {n_replicates}\n")
            f.write(
                f"Training mode       : {'Single-task' if task_name else f'Multi-task ({len(self.categories)} bioactivities)'}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Output directory    : {outputdir}\n")

        print("\n" + "=" * 80)
        print("Hold-out Test Mean (MCC ± SEM)".center(80))
        print("=" * 80)
        print(f"{'Task':<8} {'Mean MCC':<10} ± SEM")
        print("-" * 80)
        for name, m, s in zip(task_names_for_dict, per_task_mean, per_task_sem):
            print(f"{name:<8} {m:8.4f}  ± {s:.4f}")
        print("-" * 80)
        print(f"{'Overall':<8} {overall_mean:8.4f}  ± {overall_sem:.4f}")

        plot_results = self.plot_mtl_performance(
            best_task_mccs=per_task_mean.tolist(),
            best_mcc_final=overall_mean,
            best_epoch="10-fold Ensemble + SVM + KNN",
            outputdir=outputdir,
            arch_name="PepMTL-Hybrid",
            hpo_name="Dual HPO + 10-fold Soft Voting + SVM+KNN",
            n_runs=n_replicates,
            sem_test_mcc=overall_sem,
            task_sem=per_task_sem.tolist(),
            task_name=task_name,
            is_holdout=True
        )

        pipeline_results = {
            'mean_mcc_test': float(overall_mean),
            'sem_mcc_test': float(overall_sem),
            'per_task_mcc': per_task_mcc_dict,
            'per_task_mean': per_task_mean_dict,
            'per_task_sem': per_task_sem_dict,
            'ensemble_used': True,
            'replicates': n_replicates,
            'plot_files': plot_results,
            'hybrid_used': True
        }

        return pipeline_results

    def train_bilstm(self,
                     df_train: pd.DataFrame,
                     df_test: pd.DataFrame,
                     train_emb: np.ndarray,
                     test_emb: np.ndarray,
                     outputdir: str,
                     task_name: str = None,
                     hpo_strategy: str = "dual",
                     hpo_trials: int = 10,
                     max_epochs_per_trial: int = 100,
                     final_epochs: int = 200,
                     seed: int = 42,
                     n_replicates: int = 3,
                     **kwargs) -> dict:
        """
        Train a pure Bi-LSTM Multi-Task Learning (MTL) model.
        Args:
            df_train: Metadata DataFrame containing training target labels.
            df_test: Metadata DataFrame containing testing target labels.
            train_emb: Precomputed feature embeddings for the training set.
            test_emb: Precomputed feature embeddings for the testing set.
            outputdir: Directory path to save model checkpoints, logs, and evaluation plots.
            task_name: Name of a specific task if running single-task training,
                or None for multi-task training across all categories. Defaults to None.
            hpo_strategy: Strategy for hyperparameter optimization. Defaults to "dual".
            hpo_trials: Number of optimization trials per sampler. Defaults to 10.
            max_epochs_per_trial: Maximum training epochs allowed per HPO trial. Defaults to 100.
            final_epochs: Epoch limit for training final ensemble models. Defaults to 200.
            seed: Base random seed for reproducibility across replicates. Defaults to 42.
            n_replicates: Number of independent experimental replicates. Defaults to 3.
        Returns:
            dict: A structured dictionary containing overall mean test MCC, SEM, per-task metric dictionaries,
                ensemble status, and output plot file paths.
        """
        print(f"\nStarting Bi-LSTM MTL Training...")

        num_tasks = 1 if task_name is not None else 16
        batch_size = 1024
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.final_epochs = final_epochs
        self.batch_size = batch_size
        self.pos_weight_tensor = None

        if task_name is not None:
            y_cols = [f'Y_{task_name}']
            task_names = [task_name]
        else:
            y_cols = [f'Y_{cat}' for cat in self.categories]
            task_names = self.categories

        replicate_overall_mccs = []
        replicate_per_task_mccs = []

        for rep in range(n_replicates):
            print(f"\nReplicate {rep + 1}/{n_replicates} (seed={seed + rep * 100})")
            current_seed = seed + rep * 100
            self.current_run_idx = rep

            x_train_np = train_emb
            y_train_np = df_train[y_cols].values
            x_test_np = test_emb
            y_test_np = df_test[y_cols].values

            # Split training set for validation evaluation and positive class weighting calculation
            x_tr, x_val, y_tr, y_val = train_test_split(x_train_np, y_train_np, test_size=0.1,
                                                        random_state=current_seed)
            x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32).to(device)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
            y_val_np = y_val

            # Compute positive class loss scaling weights
            pos_counts_np = np.sum(y_tr, axis=0).astype(np.float32)
            neg_counts_np = len(y_tr) - pos_counts_np
            pos_ratio_np = neg_counts_np / np.clip(pos_counts_np, 1.0, None)
            pos_weights_np = np.clip(pos_ratio_np, 1.0, 140.0)
            self.pos_weight_tensor = torch.tensor(pos_weights_np, dtype=torch.float32, device=device)

            # Hyperparameter Optimization
            def objective(trial):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                hparams = {
                    'lstm_num_layers': trial.suggest_int('lstm_num_layers', 2, 3),
                    'lstm_hidden': trial.suggest_categorical('lstm_hidden', [512, 768, 1024]),
                    'head_hidden': trial.suggest_categorical('head_hidden', [384, 512, 768]),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                    'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 8e-4, log=True),
                }

                model = self.create_model_bilsm(hparams, x_train_np.shape[1], num_tasks)
                optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'],
                                              weight_decay=hparams['weight_decay'])
                criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
                scaler = torch.amp.GradScaler('cuda')

                best_val_mcc = -1.0
                patience_cnt = 0
                patience = 10
                min_delta = 0.001

                for epoch in tqdm(range(max_epochs_per_trial),
                                  desc=f"Trial {trial.number} Epochs",
                                  unit="epoch",
                                  leave=False,
                                  mininterval=2):
                    model.train()
                    perm = torch.randperm(len(x_tr_tensor))
                    for i in range(0, len(x_tr_tensor), batch_size):
                        idx = perm[i:i + batch_size]
                        x_batch = x_tr_tensor[idx]
                        y_batch = y_tr_tensor[idx]

                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda'):
                            task_pred, _ = model(x_batch)
                            loss = criterion(task_pred, y_batch)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    val_mcc = self.evaluate_model(
                        model=model,
                        x_data=x_val_tensor,
                        y_data=y_val_np,
                        batch_size=batch_size,
                        device=device,
                        num_tasks=num_tasks
                    )

                    if val_mcc > best_val_mcc + min_delta:
                        best_val_mcc = val_mcc
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

                return best_val_mcc

            hpo_results = {}

            print("\nRunning Random Search HPO:")
            study_random = optuna.create_study(direction="maximize", sampler=RandomSampler(current_seed))
            with tqdm(total=hpo_trials, desc="Random HPO", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix(best=f"{study.best_value:.4f}")

                study_random.optimize(objective, n_trials=hpo_trials, callbacks=[callback])

            hpo_results['Random'] = {
                'best_value': study_random.best_value,
                'best_params': study_random.best_params,
                'n_trials': len(study_random.trials)
            }

            print("\nRunning Bayesian Optimization HPO:")
            study_tpe = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(
                    seed=current_seed,
                    multivariate=True,
                    group=True
                )
            )
            with tqdm(total=hpo_trials, desc="Bayesian HPO", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix(best=f"{study.best_value:.4f}")

                study_tpe.optimize(objective, n_trials=hpo_trials, callbacks=[callback])

            hpo_results['Bayesian'] = {
                'best_value': study_tpe.best_value,
                'best_params': study_tpe.best_params,
                'n_trials': len(study_tpe.trials)
            }

            # Compare and select optimal HPO sampler strategy
            winner = 'Random' if hpo_results['Random']['best_value'] > hpo_results['Bayesian'][
                'best_value'] else 'Bayesian'
            best_params = study_tpe.best_params if winner == 'Bayesian' else study_random.best_params

            print("\n" + "=" * 80)
            title = f"HPO Method Comparison (each method runs {hpo_trials} trials)"
            print(title.center(80))
            print("=" * 80)

            col_widths = [18, 20, 10, 22]
            headers = [
                "Method".center(col_widths[0]),
                "Best MCC".center(col_widths[1]),
                "Trials".center(col_widths[2]),
                "Recommendation".center(col_widths[3])
            ]
            print("".join(headers))
            print("-" * 80)

            for method in ['Random', 'Bayesian']:
                res = hpo_results[method]
                recommend = " Recommended" if method == winner else ""
                row = [
                    method.center(col_widths[0]),
                    f"{res['best_value']:.4f}".center(col_widths[1]),
                    str(res['n_trials']).center(col_widths[2]),
                    recommend.center(col_widths[3])
                ]
                line = "".join(row)
                if method == winner:
                    print(f"\033[1m{line}\033[0m")
                else:
                    print(line)

            print("-" * 80)
            diff = abs(hpo_results['Random']['best_value'] - hpo_results['Bayesian']['best_value'])
            conclusion = f"Conclusion: {winner} performs better (MCC higher by {diff:.4f})"
            print(conclusion.center(80))

            print(f"\nUsing the best parameters from {winner} for final training:")
            print(str(best_params))

            # === 呼叫 ensemble（不變）===
            ensemble_mcc, per_task_mccs, task_names, _ = self.ensemble(
                best_params, x_train_np, y_train_np, x_test_np, y_test_np,
                outputdir, current_seed, num_tasks
            )
            replicate_overall_mccs.append(ensemble_mcc)
            replicate_per_task_mccs.append(per_task_mccs)

        # Aggregate Results across Experimental Replicates
        replicate_per_task_mccs = np.array(replicate_per_task_mccs)
        per_task_mean = np.nanmean(replicate_per_task_mccs, axis=0)
        per_task_sem = np.nanstd(replicate_per_task_mccs, axis=0, ddof=1) / np.sqrt(n_replicates)

        overall_mean = np.nanmean(per_task_mean)
        overall_sem = np.std(replicate_overall_mccs, ddof=1) / np.sqrt(n_replicates)

        # Format result dictionaries
        task_names_for_dict = task_names  # 最後一次 ensemble 已經回傳正確的 task_names
        per_task_mcc_dict = dict(zip(task_names_for_dict, per_task_mean.tolist()))
        per_task_mean_dict = dict(zip(task_names_for_dict, per_task_mean.tolist()))
        per_task_sem_dict = dict(zip(task_names_for_dict, per_task_sem.tolist()))

        print("\n" + "=" * 80)
        print("Hold-out Test Mean (MCC ± SEM)".center(40))
        print("=" * 80)
        print(f"{'Task':<8} {'Mean MCC':<10} ± SEM")
        print("-" * 80)
        for name, m, s in zip(task_names_for_dict, per_task_mean, per_task_sem):
            print(f"{name:<8} {m:8.4f}  ± {s:.4f}")
        print("-" * 80)
        print(f"{'Overall':<8} {overall_mean:8.4f}  ± {overall_sem:.4f}")

        # Generate MTL performance evaluation plots
        plot_results = self.plot_mtl_performance(
            best_task_mccs=per_task_mean.tolist(),
            best_mcc_final=overall_mean,
            best_epoch="10-fold Ensemble",
            outputdir=outputdir,
            arch_name="PepMTL",
            hpo_name="Dual HPO + 10-fold Soft Voting",
            n_runs=n_replicates,
            sem_test_mcc=overall_sem,
            task_sem=per_task_sem.tolist(),
            task_name=task_name,
            is_holdout=True
        )

        # Construct final evaluation summary dictionary
        pipeline_results = {
            'mean_mcc_test': float(overall_mean),
            'sem_mcc_test': float(overall_sem),
            'per_task_mcc': per_task_mcc_dict,
            'per_task_mean': per_task_mean_dict,
            'per_task_sem': per_task_sem_dict,
            'ensemble_used': True,
            'replicates': n_replicates,
            'plot_files': plot_results
        }

        return pipeline_results

    def train_cnn(self,
                  df_train: pd.DataFrame,
                  df_test: pd.DataFrame,
                  train_emb: np.ndarray,
                  test_emb: np.ndarray,
                  outputdir: str,
                  task_name: str = None,
                  hpo_strategy: str = "dual",
                  hpo_trials: int = 10,
                  max_epochs_per_trial: int = 100,
                  final_epochs: int = 200,
                  seed: int = 42,
                  n_replicates: int = 3,
                  **kwargs) -> dict:
        """
        Train a 1D CNN Multi-Task Learning (MTL) model.
        Args:
            df_train: Metadata DataFrame containing training target labels.
            df_test: Metadata DataFrame containing testing target labels.
            train_emb: Precomputed feature embeddings for the training set.
            test_emb: Precomputed feature embeddings for the testing set.
            outputdir: Directory path to save model checkpoints, logs, and evaluation plots.
            task_name: Name of a specific task if running single-task training,
                or None for multi-task training across all categories. Defaults to None.
            hpo_strategy: Strategy for hyperparameter optimization. Defaults to "dual".
            hpo_trials: Number of optimization trials per sampler. Defaults to 10.
            max_epochs_per_trial: Maximum training epochs allowed per HPO trial. Defaults to 100.
            final_epochs: Epoch limit for training final ensemble models. Defaults to 200.
            seed: Base random seed for reproducibility across replicates. Defaults to 42.
            n_replicates: Number of independent experimental replicates. Defaults to 3.
        Returns:
            dict: A structured dictionary containing overall mean test MCC, SEM, per-task metric dictionaries,
                ensemble status, and output plot file paths.
        """
        print(f"\nStarting CNN MTL Training...")

        num_tasks = 1 if task_name is not None else 16
        batch_size = 1024
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.final_epochs = final_epochs
        self.batch_size = batch_size
        self.pos_weight_tensor = None

        if task_name is not None:
            y_cols = [f'Y_{task_name}']
            task_names = [task_name]
        else:
            y_cols = [f'Y_{cat}' for cat in self.categories]
            task_names = self.categories

        replicate_overall_mccs = []
        replicate_per_task_mccs = []

        for rep in range(n_replicates):
            print(f"\nReplicate {rep + 1}/{n_replicates} (seed={seed + rep * 100})")
            current_seed = seed + rep * 100
            self.current_run_idx = rep

            x_train_np = train_emb
            y_train_np = df_train[y_cols].values
            x_test_np = test_emb
            y_test_np = df_test[y_cols].values

            # Split training set for validation evaluation and positive class weighting calculation
            x_tr, x_val, y_tr, y_val = train_test_split(x_train_np, y_train_np, test_size=0.1,
                                                        random_state=current_seed)
            x_tr_tensor = torch.tensor(x_tr, dtype=torch.float32).to(device)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
            y_val_np = y_val

            pos_counts_np = np.sum(y_tr, axis=0).astype(np.float32)
            neg_counts_np = len(y_tr) - pos_counts_np
            pos_ratio_np = neg_counts_np / np.clip(pos_counts_np, 1.0, None)
            pos_weights_np = np.clip(pos_ratio_np, 1.0, 140.0)
            self.pos_weight_tensor = torch.tensor(pos_weights_np, dtype=torch.float32, device=device)

            # Hyperparameter Optimization
            def objective(trial):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                hparams = {
                    'cnn_out': trial.suggest_categorical('cnn_out', [384, 512, 768]),   #[384, 512, 768]
                    'cnn_kernel': trial.suggest_int('cnn_kernel', 3, 5),
                    'head_hidden': trial.suggest_categorical('head_hidden', [384, 512, 768]),   #[384, 512, 768]
                    'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                    'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 8e-4, log=True),
                }

                model = self.create_model_cnn(hparams, x_train_np.shape[1], num_tasks)

                # Wrap with DataParallel if multiple GPU devices are available
                if torch.cuda.device_count() > 1:
                    #print(f"Use {torch.cuda.device_count()} GPU (DataParallel)")
                    model = torch.nn.DataParallel(model)

                model = model.to(self.device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'],
                                              weight_decay=hparams['weight_decay'])
                criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
                scaler = torch.amp.GradScaler('cuda')

                best_val_mcc = -1.0
                patience_cnt = 0
                patience = 10
                min_delta = 0.001

                for epoch in tqdm(range(max_epochs_per_trial),
                                  desc=f"Trial {trial.number} Epochs",
                                  unit="epoch",
                                  leave=False,
                                  mininterval=2):
                    model.train()
                    perm = torch.randperm(len(x_tr_tensor))
                    for i in range(0, len(x_tr_tensor), batch_size):
                        idx = perm[i:i + batch_size]
                        x_batch = x_tr_tensor[idx]
                        y_batch = y_tr_tensor[idx]

                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda'):
                            task_pred, _ = model(x_batch)
                            loss = criterion(task_pred, y_batch)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    val_mcc = self.evaluate_model(
                        model=model,
                        x_data=x_val_tensor,
                        y_data=y_val_np,
                        batch_size=batch_size,
                        device=device,
                        num_tasks=num_tasks
                    )

                    if val_mcc > best_val_mcc + min_delta:
                        best_val_mcc = val_mcc
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

                return best_val_mcc

            # Execute Dual HPO (Random Search + Bayesian Optimization)
            print("\nRunning Random Search HPO:")
            study_random = optuna.create_study(direction="maximize", sampler=RandomSampler(current_seed))
            with tqdm(total=hpo_trials, desc="Random HPO", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix(best=f"{study.best_value:.4f}")

                study_random.optimize(objective, n_trials=hpo_trials, callbacks=[callback])

            hpo_results = {}
            hpo_results['Random'] = {
                'best_value': study_random.best_value,
                'best_params': study_random.best_params,
                'n_trials': len(study_random.trials)
            }

            print("\nRunning Bayesian Optimization HPO:")
            study_tpe = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=current_seed, multivariate=True, group=True)
            )
            with tqdm(total=hpo_trials, desc="Bayesian HPO", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix(best=f"{study.best_value:.4f}")

                study_tpe.optimize(objective, n_trials=hpo_trials, callbacks=[callback])

            hpo_results['Bayesian'] = {
                'best_value': study_tpe.best_value,
                'best_params': study_tpe.best_params,
                'n_trials': len(study_tpe.trials)
            }

            # HPO Method Comparison
            winner = 'Random' if hpo_results['Random']['best_value'] > hpo_results['Bayesian'][
                'best_value'] else 'Bayesian'
            best_params = study_tpe.best_params if winner == 'Bayesian' else study_random.best_params

            print("\n" + "=" * 80)
            title = f"HPO Method Comparison (each method runs {hpo_trials} trials)"
            print(title.center(80))
            print("=" * 80)

            col_widths = [18, 20, 10, 22]
            headers = [
                "Method".center(col_widths[0]),
                "Best MCC".center(col_widths[1]),
                "Trials".center(col_widths[2]),
                "Recommendation".center(col_widths[3])
            ]
            print("".join(headers))
            print("-" * 80)

            for method in ['Random', 'Bayesian']:
                res = hpo_results[method]
                recommend = " Recommended" if method == winner else ""
                row = [
                    method.center(col_widths[0]),
                    f"{res['best_value']:.4f}".center(col_widths[1]),
                    str(res['n_trials']).center(col_widths[2]),
                    recommend.center(col_widths[3])
                ]
                line = "".join(row)
                if method == winner:
                    print(f"\033[1m{line}\033[0m")
                else:
                    print(line)

            print("-" * 80)
            diff = abs(hpo_results['Random']['best_value'] - hpo_results['Bayesian']['best_value'])
            conclusion = f"Conclusion: {winner} performs better (MCC higher by {diff:.4f})"
            print(conclusion.center(80))

            print(f"\nUsing the best parameters from {winner} for final training:")
            print(str(best_params))

            # Execute 10-fold CV Soft-Voting Ensemble
            ensemble_mcc, per_task_mccs, task_names, _ = self.ensemble(
                best_params, x_train_np, y_train_np, x_test_np, y_test_np,
                outputdir, current_seed, num_tasks
            )
            replicate_overall_mccs.append(ensemble_mcc)
            replicate_per_task_mccs.append(per_task_mccs)

        # Aggregate Results across Experimental Replicates
        replicate_per_task_mccs = np.array(replicate_per_task_mccs)
        per_task_mean = np.nanmean(replicate_per_task_mccs, axis=0)
        per_task_sem = np.nanstd(replicate_per_task_mccs, axis=0, ddof=1) / np.sqrt(n_replicates)

        overall_mean = np.nanmean(per_task_mean)
        overall_sem = np.std(replicate_overall_mccs, ddof=1) / np.sqrt(n_replicates)

        task_names_for_dict = task_names
        per_task_mcc_dict = dict(zip(task_names_for_dict, per_task_mean.tolist()))
        per_task_mean_dict = dict(zip(task_names_for_dict, per_task_mean.tolist()))
        per_task_sem_dict = dict(zip(task_names_for_dict, per_task_sem.tolist()))

        print("\n" + "=" * 40)
        print("Hold-out Test Mean (MCC ± SEM)".center(40))
        print("=" * 40)
        print(f"{'Task':<8} {'Mean MCC':<10} ± SEM")
        print("-" * 40)
        for name, m, s in zip(task_names_for_dict, per_task_mean, per_task_sem):
            print(f"{name:<8} {m:8.4f}  ± {s:.4f}")
        print("-" * 40)
        print(f"{'Overall':<8} {overall_mean:8.4f}  ± {overall_sem:.4f}")

        # Generate MTL performance evaluation plots
        plot_results = self.plot_mtl_performance(
            best_task_mccs=per_task_mean.tolist(),
            best_mcc_final=overall_mean,
            best_epoch="10-fold Ensemble",
            outputdir=outputdir,
            arch_name="PepMTL-CNN",
            hpo_name="Dual HPO + 10-fold Soft Voting",
            n_runs=n_replicates,
            sem_test_mcc=overall_sem,
            task_sem=per_task_sem.tolist(),
            task_name=task_name,
            is_holdout=True
        )

        # Construct final evaluation summary dictionary
        pipeline_results = {
            'mean_mcc_test': float(overall_mean),
            'sem_mcc_test': float(overall_sem),
            'per_task_mcc': per_task_mcc_dict,
            'per_task_mean': per_task_mean_dict,
            'per_task_sem': per_task_sem_dict,
            'ensemble_used': True,
            'replicates': n_replicates,
            'plot_files': plot_results
        }

        return pipeline_results

    def plot_mtl_performance(
            self,
            best_task_mccs: List[float],
            best_mcc_final: float,
            best_epoch: Union[int, str],
            outputdir: str,
            arch_name: str = "Unknown",
            hpo_name: str = "Manual",
            n_runs: int = 1,
            sem_test_mcc: float = 0.0,
            task_sem: List[float] = None,
            task_name: str = None,
            is_holdout: bool = True
            ) -> Dict[str, str]:
        """
        Generate publication-ready visualizations for Multi-Task Learning performance
        Args:
            best_task_mccs: List of MCC values for each of the 16 tasks
            best_mcc_final: The global average MCC across all valid tasks (hold-out test preferred)
            best_epoch: The training epoch where the best performance was achieved (or "N/A")
            outputdir: Directory to save the PDF and PNG results
            arch_name: Name of the model architecture
            hpo_name: The Hyperparameter Optimization strategy used
            n_runs: Number of independent runs used to calculate mean and SEM (default 3)
            sem_test_mcc: Standard error of the mean (SEM) for the overall average MCC
            task_sem: List of SEM values for each individual task (used for error bars in barplot)
            task_name: Optional single task name (e.g., "AMP"). If specified, runs single-task mode with only Y_{task_name}. None for multi-task (all 16 tasks).
            is_holdout: Whether this is hold-out test result (for title and interpretation)
        Returns:
            Dict[str, str]: Dictionary containing local paths to all saved figure assets
        """
        os.makedirs(outputdir, exist_ok=True)
        print("\nGenerating figures...")

        # Create a clean heatmap showing MCC for each task
        plt.figure(figsize=(15, 5))
        mcc_array = np.array(best_task_mccs).reshape(1, -1)

        if len(best_task_mccs) == 1 and task_name is not None:
            columns = [task_name]
            yticklabels = [f'{arch_name}']
        else:
            columns = self.categories
            yticklabels = [f'{arch_name}']

        df_heat = pd.DataFrame(mcc_array, columns=columns)

        sns.heatmap(
            df_heat,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            linewidths=1.2,
            linecolor='white',
            cbar_kws={'label': 'MCC', 'shrink': 0.8},
            xticklabels=columns,
            yticklabels=yticklabels,
            annot_kws={'size': 11, 'weight': 'bold'}
        )
        plt.title(
            f'Hold-out Test Performance Heatmap\n'
            f'{"Mean" if n_runs > 1 else "Best"} Avg MCC = {best_mcc_final:.4f} '
            f'{f"± {sem_test_mcc:.4f}" if n_runs > 1 else f"@ epoch {best_epoch}"}',
            fontsize=16, fontweight='bold', pad=25
        )
        plt.ylabel('Matthews Correlation Coefficient (MCC)', fontsize=15, fontweight='bold')
        plt.xlabel('Bioactivity Task', fontsize=15, fontweight='bold')

        heatmap_pdf = os.path.join(outputdir, "mtl_performance_heatmap.pdf")
        heatmap_png = heatmap_pdf.replace('.pdf', '.png')
        plt.savefig(heatmap_pdf, dpi=400, bbox_inches='tight')
        plt.savefig(heatmap_png, dpi=400, bbox_inches='tight')
        plt.close()

        # Create a sorted bar plot with error bars (SEM) for each task
        plt.figure(figsize=(16, 9))

        # Handle single-task case (only one task)
        if len(best_task_mccs) == 1 and task_name is not None:
            tasks = [task_name]
            mccs = best_task_mccs
            # Handle SEM for single task
            if task_sem is None or len(task_sem) == 0:
                task_sem = [0.0]
            elif len(task_sem) == 1:
                task_sem = task_sem
            else:
                task_sem = task_sem[:1]
        else:
            # Multi-task: sort by MCC descending
            valid_pairs = [(cat, mcc) for cat, mcc in zip(self.categories, best_task_mccs) if not np.isnan(mcc)]
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
            tasks, mccs = zip(*valid_pairs) if valid_pairs else ([], [])
            # Prepare corresponding SEM values
            task_sem = [task_sem[i] for i, (cat, mcc) in enumerate(zip(self.categories, best_task_mccs)) if
                        not np.isnan(mcc)] if task_sem is not None else np.zeros(len(mccs))

        colors = ['#d62728' if m >= 0.70 else '#ff7f0e' if m >= 0.50 else '#1f77b4' for m in mccs]

        bars = plt.bar(
            tasks,
            mccs,
            yerr=task_sem,
            capsize=6,
            color=colors,
            edgecolor='black',
            linewidth=1.0,
            width=0.6,
            error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'alpha': 0.8}
        )

        # Label each bar with mean ± SEM value
        for bar, mcc, sem in zip(bars, mccs, task_sem):
            if sem > 0:
                text = f'{mcc:.4f} ± {sem:.4f}'
            else:
                text = f'{mcc:.4f}'
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                text,
                ha='center', va='bottom',
                fontsize=8
            )

        # Reference lines
        plt.axhline(0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.9, label='Excellent (>0.7)')
        plt.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Good (0.5 - 0.7)')
        plt.axhline(0.25, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Fair (0.25 - 0.5)')
        plt.axhspan(0, 0.25, facecolor='lightgray', alpha=0.2, label='Poor (<0.25)')
        plt.ylabel('Matthews Correlation Coefficient (MCC)', fontsize=15, fontweight='bold')
        plt.xlabel('Bioactivity Task', fontsize=15, fontweight='bold')
        plt.title(
            f'Hold-out Test Performance ({arch_name} + {hpo_name})\n'
            f'{"Mean" if n_runs > 1 else "Best"} Average MCC = {best_mcc_final:.4f} '
            f'{f"± {sem_test_mcc:.4f}" if n_runs > 1 else f"@ epoch {best_epoch}"}',
            fontsize=18, fontweight='bold', pad=30
        )

        plt.ylim(0, max(mccs, default=0.3) * 1.25)
        plt.grid(axis='y', alpha=0.35, linestyle='--', zorder=0)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.tight_layout()

        bar_pdf = os.path.join(outputdir, "mtl_per_task_mcc_barplot.pdf")
        bar_png = bar_pdf.replace('.pdf', '.png')
        plt.savefig(bar_pdf, dpi=400, bbox_inches='tight')
        plt.savefig(bar_png, dpi=400, bbox_inches='tight')
        plt.close()

        # Generate a text file that matches the style of AutoPeptideML paper summary
        summary_text = f"""
        AutoPeptideML-style Output Summary

        1. Introduction
        This document summarizes the performance of {arch_name} model trained with {hpo_name} HPO.
        Results are based on {'hold-out test set' if is_holdout else '10-fold CV'}.
        General guideline: MCC between 0.25-0.5 indicates low correlation; model predictions may not be highly reliable.
        The actual acceptability depends on your target application and state-of-the-art.

        2. Main Performance Metrics
        - Overall {'Mean' if n_runs > 1 else 'Best'} MCC: {best_mcc_final:.4f} {'± SEM' if n_runs > 1 else ''} {f'± {sem_test_mcc:.4f}' if n_runs > 1 else f'@ epoch {best_epoch}'}

        3. Interpretation of MCC
        - > 0.7: Excellent performance (strong correlation, reliable predictions)
        - 0.5 - 0.7: Good performance (moderate correlation, useful in many contexts)
        - 0.25 - 0.5: Fair/low correlation (predictions have some value but limited reliability)
        - < 0.25: Poor or random performance (not recommended for most applications)

        4. Confusion Matrix Concept (for reference)
        - True Negative (TN): Negative samples correctly predicted as negative.
        - False Positive (FP): Negative samples incorrectly predicted as positive (over-sensitive model).
        - False Negative (FN): Positive samples incorrectly predicted as negative (highly specific model).
        - True Positive (TP): Positive samples correctly predicted as positive.

        Note: Confusion matrix and per-task detailed metrics are available in the pipeline JSON results.
        """
        summary_txt = os.path.join(outputdir, "performance_summary.txt")
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print(f"Heatmap saved to: {heatmap_pdf}")
        print(f"Bar plot saved to: {bar_pdf}")
        print(f"Performance summary text saved to: {summary_txt}")

        return {
            'heatmap_pdf': heatmap_pdf,
            'heatmap_png': heatmap_png,
            'barplot_pdf': bar_pdf,
            'barplot_png': bar_png,
            'summary_txt': summary_txt,
        }

    def run_full_pipeline(
            self,
            positives_dir: str = None,
            outputdir: str = None,
            autosearch: str = 'auto',
            autosearch_tags: str = '',
            neg_ratio: float = 1.0,
            test_size: float = 0.2,
            similarity_threshold: float = 0.3,
            train_model_flag: bool = True,
            model_name: str = "esm2_t33_650M_UR50D",
            reuse_data: str = None,
            task_name: str = None,
    ) -> Dict[str, Union[str, float]]:
        """Execute the end-to-end multi-task/single-task peptide property prediction pipeline.
            Args:
                positives_dir: Directory containing FASTA files of positive sequences. Defaults to None.
                outputdir: Directory path to store intermediate datasets, precomputed
                    embeddings, checkpoints, and performance logs. Defaults to None.
                autosearch: Mode for automated negative sequence mining. Defaults to 'auto'.
                autosearch_tags: Comma-separated tags used for negative retrieval. Defaults to ''.
                neg_ratio: Ratio of negative samples to retrieve relative to positive sequences. Defaults to 1.0.
                test_size: Fraction of the dataset to allocate to the hold-out test set. Defaults to 0.2.
                similarity_threshold: Sequence identity threshold for CD-HIT homology clustering. Defaults to 0.3.
                train_model_flag: Whether to execute the model training phase after data preparation. Defaults to True.
                model_name: Name or HuggingFace path of the pre-trained ESM-2 model. Defaults to "esm2_t33_650M_UR50D".
                reuse_data: Directory containing preprocessed CSV datasets and precomputed
                    `.npy` embeddings to bypass Steps 1–4. Defaults to None.
                task_name: Name of a specific bioactivity category for single-task mode,
                    or None to execute multi-task mode across all target categories. Defaults to None.
            Returns:
                Dict: Dictionary containing training execution status, file paths,
                    or multi-task evaluation metrics depending on pipeline configuration.
            Raises:
                ValueError: If `task_name` is provided but not contained within `self.categories`.
                FileNotFoundError: If `reuse_data` is specified but missing required dataset or embedding files.
            """

        os.makedirs(outputdir, exist_ok=True)

        if task_name is not None:
            if task_name not in self.categories:
                raise ValueError(f"Invalid task_name '{task_name}', available: {self.categories}")
            y_cols = [f'Y_{task_name}']
            num_tasks = 1
        else:
            y_cols = [f'Y_{cat}' for cat in self.categories]
            num_tasks = len(y_cols)

        # Reuse existing preprocessed datasets and embeddings if specified
        if reuse_data:
            reuse_dir = reuse_data
            required_files = [
                'train_dataset.csv',
                'test_dataset.csv',
                'train_embeddings.npy',
                'test_embeddings.npy'
            ]
            missing = [f for f in required_files if not os.path.exists(os.path.join(reuse_dir, f))]
            if missing:
                raise FileNotFoundError(f"[Reuse Error] Missing file: {missing}\n")

            print(f"Resuing data from: {reuse_dir}")
            print("\nSkipping Step 1 to 4 (Data prep + Negative search + Partition + Embedding)")
            #print("Directly to Step 5: Model Training")
            print("=" * 80)

            df_train = pd.read_csv(os.path.join(reuse_dir, 'train_dataset.csv'))
            df_test = pd.read_csv(os.path.join(reuse_dir, 'test_dataset.csv'))
            train_emb = np.load(os.path.join(reuse_dir, 'train_embeddings.npy'))
            test_emb = np.load(os.path.join(reuse_dir, 'test_embeddings.npy'))

        else:
            # Step 1: Data Curation
            merged_path = os.path.join(outputdir, "merged_positives.csv")
            if os.path.exists(merged_path):
                if self.verbose:
                    print("\nStep 1: Data Curation")
                    print("Found existing merged_positives.csv → Skipping Step 1 (Data Curation)")
                df = pd.read_csv(merged_path)
                print(f"Loaded {len(df):,} sequences from {merged_path}")
            else:
                if self.verbose:
                    print("\nStep 1: Data Curation")
                df = self.curate_dataset(positives_dir, outputdir)

            # Step 2: Negative Sample Searching
            if self.verbose:
                print("\nStep 2: Negative Sample Searching")
            if self.verbose:
                print(f"Adding negative samples (ratio = 1:{int(neg_ratio)} based on positive sequences)...")

            df = self.autosearch_negatives(
                df_pos=df,
                positive_tags=[t.strip() for t in autosearch_tags.split(',') if t.strip()],
                proportion=neg_ratio,
                save=True,
                autosearch=autosearch,
                outputdir=outputdir
            )

            full_path = os.path.join(outputdir, "full_dataset.csv")
            df.to_csv(full_path, index=False)

            n_sequences = len(df)
            n_positive_seq = (df[y_cols].sum(axis=1) > 0).sum()
            n_negative_seq = n_sequences - n_positive_seq
            n_positive_labels = int(df[y_cols].sum().sum())
            n_negative_labels = n_sequences * len(y_cols) - n_positive_labels

            if self.verbose:
                print(f"Dataset saved to: {full_path}")
                print(f"  - Total sequences       : {n_sequences:,}")
                print(f"  - Positive sequences    : {n_positive_seq:,}")
                print(f"  - Pure negative sequences: {n_negative_seq:,}")
                print(f"  - Total positive labels : {n_positive_labels:,}")
                print(f"  - Total negative labels : {n_negative_labels:,}")
                print(f"  - Avg labels/sequence   : {n_positive_labels / n_sequences:.4f}")

            # Step 3: Homology Partitioning
            if self.verbose:
                print("\nStep 3: Homology-Based Partitioning")

            df_train, df_test = self.homology_partitioning(df,
                                                           test_size=test_size,
                                                           similarity_threshold=similarity_threshold
                                                           )
            train_path = os.path.join(outputdir, 'train_dataset.csv')
            test_path = os.path.join(outputdir, 'test_dataset.csv')
            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path, index=False)

            stats = {
                'total_sequences': len(df),
                'total_positive_labels': int(df[y_cols].sum().sum()),
                'total_negative_labels': len(df) * len(y_cols) - df[y_cols].sum().sum(),
                'train_size': len(df_train),
                'test_size': len(df_test),
            }
            stats_path = os.path.join(outputdir, 'dataset_statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(
                    stats,
                    f,
                    indent=4,
                    default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o,
                                                                                                      np.floating) else o
                )
            results = {
                'train_dataset': train_path,
                'test_dataset': test_path,
                'statistics': stats_path,
                'train_size': len(df_train),
                'test_size': len(df_test)
            }

            # Step 4: ESM-2 Embedding Generation
            if train_model_flag:
                if self.verbose:
                    print(f"\nStep 4: ESM-2 Embedding Generation (using {model_name})")

                train_embeddings = self.generate_esm_embeddings(df_train,
                                                                model_name=model_name,
                                                                outputdir=outputdir)
                test_embeddings = self.generate_esm_embeddings(df_test,
                                                               model_name=model_name,
                                                               outputdir=outputdir)

                train_emb_path = os.path.join(outputdir, 'train_embeddings.npy')
                test_emb_path = os.path.join(outputdir, 'test_embeddings.npy')
                np.save(train_emb_path, train_embeddings)
                np.save(test_emb_path, test_embeddings)

                results['train_embeddings'] = train_emb_path
                results['test_embeddings'] = test_emb_path

                train_emb = np.load(train_emb_path)
                test_emb = np.load(test_emb_path)
            else:
                train_emb = test_emb = None

        # Step 5: Model Training (single task/ multi-tasks)
        print("\nStep 5:  Model Training (single task/ multi-tasks)")

        if not train_model_flag:
            print("train_model_flag=False → Skipping training phase.")
            return {'status': 'training_skipped', 'outputdir': outputdir}

        pipeline_results = self.train_model(
            df_train=df_train,
            df_test=df_test,
            train_emb=train_emb,
            test_emb=test_emb,
            outputdir=outputdir,
            task_name=task_name,
            hpo_trials=10,
            n_runs=3,
        )

        return pipeline_results