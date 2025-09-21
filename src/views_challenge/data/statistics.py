from joblib import Parallel
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

from views_pipeline_core.data.handlers import PGMDataset


def calculate_threshold_probabilities(
    dataset: PGMDataset, thresholds: list, features: Optional[list] = None
) -> pd.DataFrame:
    """
    Calculate exceedance probabilities for prediction distributions efficiently.

    Args:
        dataset (PGMDataset): The PGM dataset from views_pipeline_core
        thresholds (list): List of numeric thresholds (0–1 for probability fractions)
        features (list, optional): List of variables to calculate probabilities for.
                                   Defaults to all targets.

    Returns:
        pd.DataFrame: Multi-index (time, entity) DataFrame with columns
                      like "{var}_p>{threshold}" for each feature and threshold.
    """
    selected_vars = features or dataset.targets
    tensor = dataset.to_tensor()  # (time, entity, samples, vars)
    n_time, n_entity, n_samples, _ = tensor.shape
    results = []

    for var_name in tqdm(selected_vars, desc="Processing features"):
        var_idx = dataset.targets.index(var_name)
        var_tensor = tensor[..., var_idx]  # (time, entity, samples)
        flat_tensor = var_tensor.reshape(-1, n_samples)  # (time*entity, samples)
        n_flat = flat_tensor.shape[0]

        # Batch processing parameters
        batch_size = 1000
        batches = [
            flat_tensor[i : i + batch_size] for i in range(0, n_flat, batch_size)
        ]
        prob_flat = []

        with Parallel(n_jobs=-1, prefer="threads"):
            for batch in tqdm(batches, desc=f"{var_name} batches"):
                batch_results = np.array(
                    [[(samples > t).mean() for t in thresholds] for samples in batch]
                )
                prob_flat.append(batch_results)

        prob_array = np.vstack(prob_flat)  # shape (time*entity, n_thresholds)
        index = pd.MultiIndex.from_product(
            [
                dataset.dataframe.index.get_level_values(dataset._time_id).unique(),
                dataset.dataframe.index.get_level_values(dataset._entity_id).unique(),
            ],
            names=[dataset._time_id, dataset._entity_id],
        )
        columns = [f"{var_name}_p>{t}" for t in thresholds]
        df = pd.DataFrame(prob_array, index=index, columns=columns)
        results.append(df)

    return pd.concat(results, axis=1)
