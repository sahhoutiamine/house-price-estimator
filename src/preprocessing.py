"""
src/preprocessing.py

Shared preprocessing helpers used by the Streamlit app to turn ONE new house's
raw form values into the exact 231-column, scaled feature vector the model
was trained on in 01_data_exploration_preprocessing.ipynb.

Nothing here is retrained or refit -- it only replays the SAME steps that
notebook already did, using the artifacts it saved to models/:
    - models/scaler.pkl              (fitted StandardScaler)
    - models/feature_columns.pkl     (the exact 231 training column names, in order)
    - models/continuous_columns.pkl  (which of those columns get scaled)
    - models/default_row.pkl         (median/mode values for every raw column,
                                       used to fill in anything the user didn't
                                       type on the form)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# These constants are copied VERBATIM from
# 01_data_exploration_preprocessing.ipynb (Section 4), so a new house row goes
# through the exact same cleaning / encoding logic as the training data did.
# ---------------------------------------------------------------------------

QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORD_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

FINISH_MAP = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
EXPOSURE_MAP = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}

PORCH_COLS = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the ordinal-encoded + engineered columns, exactly like notebook 01
    Section 4 ("Ordinal Quality Encoding & Domain Feature Engineering").

    `df` is a single-row DataFrame that starts from `default_row` (already
    cleaned/imputed) with a few raw values overwritten by the user's form input.
    """
    df = df.copy()

    # Quality columns -> numbers. `default_row` already stores these as numbers
    # (notebook 01 mapped them before saving df_clean), so this only matters
    # if a raw text value ever ends up here -- kept for safety.
    for col in ORD_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(QUAL_MAP).fillna(0)

    if "GarageFinish" in df.columns and df["GarageFinish"].dtype == object:
        df["GarageFinish"] = df["GarageFinish"].map(FINISH_MAP).fillna(0)

    if "BsmtExposure" in df.columns and df["BsmtExposure"].dtype == object:
        df["BsmtExposure"] = df["BsmtExposure"].map(EXPOSURE_MAP).fillna(0)

    # Domain feature engineering -- identical formulas to notebook 01 / 03.
    df["TotalArea"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (
        df["FullBath"] + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"])
    )
    df["HouseAge"] = np.maximum(0, df["YrSold"] - df["YearBuilt"])
    df["YearsSinceRemodel"] = np.maximum(0, df["YrSold"] - df["YearRemodAdd"])
    df["TotalPorchArea"] = df[PORCH_COLS].sum(axis=1)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

    return df


def build_model_input(
    user_inputs: dict,
    default_row: pd.DataFrame,
    scaler,
    feature_columns: list,
    continuous_columns: list,
) -> pd.DataFrame:
    """
    Turn a dict of user-provided raw values into a single-row DataFrame with
    EXACTLY the same columns, order, and scaling as the model's training data.

    Steps:
      1. Start from `default_row` (median/mode of every raw column) and
         overwrite it with whatever the user filled in on the form.
      2. Recompute engineered features (TotalArea, HouseAge, ...) on this row.
      3. One-hot encode the categorical columns, then align to
         `feature_columns` so every training column exists (filled with 0
         if this specific row doesn't produce it).
      4. Scale the continuous columns with the ALREADY-FITTED scaler from
         training (`.transform()`, never `.fit_transform()`).
    """
    # 1. Start from the default row (never mutate the cached original!)
    row = default_row.copy()
    for key, value in user_inputs.items():
        row[key] = value

    # 2. Recompute engineered features on this updated single row
    row = engineer_features(row)

    # 3. One-hot encode categorical columns.
    #    IMPORTANT: drop_first must be False here. With a single row there is
    #    only ONE observed category per column, so drop_first=True would drop
    #    it every time -- silently erasing whatever the user picked (e.g. the
    #    chosen Neighborhood would never affect the prediction). Using
    #    drop_first=False and then reindexing to `feature_columns` gives the
    #    same end result the training data has, without that bug.
    nominal_cols = [c for c in row.columns if not pd.api.types.is_numeric_dtype(row[c])]
    row_encoded = pd.get_dummies(row, columns=nominal_cols, drop_first=False)

    # Add any missing training column as 0, drop anything extra, and put
    # everything back in the exact training column order.
    row_aligned = row_encoded.reindex(columns=feature_columns, fill_value=0)

    # 4. Scale the continuous columns using the scaler fitted during training
    row_aligned[continuous_columns] = scaler.transform(row_aligned[continuous_columns])

    return row_aligned
