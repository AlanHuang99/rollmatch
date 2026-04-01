"""
score_data — Compute propensity scores for matching.

Fits a logistic regression on covariates and adds a score column
(logit-transformed propensity score by default).
"""

import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression


def score_data(
    reduced_data: pl.DataFrame,
    covariates: list[str],
    treat: str,
    model_type: str = "logistic",
    match_on: str = "logit",
    max_iter: int = 1000,
) -> pl.DataFrame:
    """Fit propensity model and add scores to data.

    Parameters
    ----------
    reduced_data : pl.DataFrame
        Output from reduce_data().
    covariates : list[str]
        Column names of matching covariates.
    treat : str
        Column name for binary treatment indicator.
    model_type : str
        "logistic" (default) or "probit" (not yet supported).
    match_on : str
        "logit" for log-odds, "pscore" for raw probability.
    max_iter : int
        Maximum iterations for logistic regression.

    Returns
    -------
    pl.DataFrame
        Input data with added "score" column.
    """
    if model_type != "logistic":
        raise NotImplementedError(f"model_type='{model_type}' not supported; use 'logistic'")

    for col in covariates:
        if col not in reduced_data.columns:
            raise ValueError(f"Covariate '{col}' not found in data")

    # Extract numpy arrays for sklearn
    X = reduced_data.select(covariates).to_numpy().astype(np.float64)
    y = reduced_data[treat].to_numpy().astype(np.int32)

    # Handle NaN in covariates
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        raise ValueError(
            f"{nan_mask.sum()} rows have NaN in covariates. "
            "Remove NaN rows before scoring."
        )

    # Fit logistic regression
    model = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=42)
    model.fit(X, y)

    # Predict probabilities
    proba = model.predict_proba(X)[:, 1]  # P(treat=1)

    # Transform to requested scale
    if match_on == "logit":
        # Clip to avoid log(0) or log(inf)
        proba_clipped = np.clip(proba, 1e-10, 1 - 1e-10)
        scores = np.log(proba_clipped / (1 - proba_clipped))
    elif match_on == "pscore":
        scores = proba
    else:
        raise ValueError(f"match_on must be 'logit' or 'pscore', got '{match_on}'")

    # Add score column
    result = reduced_data.with_columns(pl.Series("score", scores))

    return result
