"""
data_cleaner.py
Cleans the raw ACIS insurance dataset and saves
a cleaned version for modeling.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))

from src.data_loader import load_and_prepare  # noqa: E402


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    In insurance data, true duplicates are data errors —
    the same policy cannot have identical records.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    print(f"   Duplicates removed: {removed:,} rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply missing value strategy based on EDA findings.

    Strategy:
    - Columns > 50% missing : drop entirely
    - Numerical < 50% missing: fill with median
    - Categorical < 50% missing: fill with mode
    - TotalClaims missing   : leave as-is (no claim = no value)
    """
    df = df.copy()

    # Step 1: Drop columns with more than 50% missing
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[
        missing_pct > 0.5
    ].index.tolist()

    # Never drop these critical columns
    protected = [
        'TotalPremium', 'TotalClaims',
        'Province', 'PolicyID'
    ]
    cols_to_drop = [
        c for c in cols_to_drop
        if c not in protected
    ]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"   Dropped {len(cols_to_drop)} high-missing "
              f"columns: {cols_to_drop}")
    else:
        print("   No columns exceeded 50% missing threshold")

    # Step 2: Fill numerical columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    filled_num = 0
    for col in num_cols:
        if col == 'TotalClaims':
            continue  # missing = no claim, leave it
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            filled_num += 1
    print(f"   Numerical columns filled with median: {filled_num}")

    # Step 3: Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    filled_cat = 0
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            filled_cat += 1
    print(f"   Categorical columns filled with mode: {filled_cat}")

    return df


def remove_outliers(df: pd.DataFrame,
                    cols: list,
                    multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove extreme outliers using the IQR method.

    We use multiplier=3.0 (instead of 1.5) because
    insurance data naturally has high-value legitimate
    claims — we only remove truly impossible values.

    Args:
        df         : DataFrame to clean
        cols       : columns to check
        multiplier : IQR multiplier (3.0 = conservative)
    """
    df = df.copy()
    total_removed = 0

    for col in cols:
        if col not in df.columns:
            print(f"   Skipping {col} — not in dataframe")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        before = len(df)
        df = df[
            (df[col].between(lower, upper)) |
            (df[col].isnull())
        ]
        removed = before - len(df)
        total_removed += removed
        print(f"   {col}: removed {removed:,} outliers "
              f"(range kept: {lower:.0f} to {upper:.0f})")

    print(f"   Total outliers removed: {total_removed:,} rows")
    return df


def clean_dataset(input_path: str,
                  output_path: str) -> pd.DataFrame:
    """
    Master cleaning function.
    Loads raw data, applies all cleaning steps,
    saves cleaned version to output_path.

    Args:
        input_path : path to raw data file
        output_path: path to save cleaned CSV

    Returns:
        df_clean: fully cleaned DataFrame
    """
    print("=" * 55)
    print("STARTING DATA CLEANING PIPELINE")
    print("=" * 55)

    # Load raw data using our existing loader
    df = load_and_prepare(input_path)
    raw_rows = df.shape[0]
    raw_cols = df.shape[1]
    print(f"\nRaw data shape: {raw_rows:,} rows, "
          f"{raw_cols} columns")

    # Step 1: Remove duplicates
    print("\n[1/3] Removing duplicates...")
    df = remove_duplicates(df)

    # Step 2: Handle missing values
    print("\n[2/3] Handling missing values...")
    df = handle_missing_values(df)

    # Step 3: Remove extreme outliers
    print("\n[3/3] Removing outliers...")
    df = remove_outliers(
        df,
        cols=[
            'TotalPremium',
            'TotalClaims',
            'CustomValueEstimate'
        ],
        multiplier=3.0
    )

    # Summary
    print("\n" + "=" * 55)
    print("CLEANING COMPLETE")
    print("=" * 55)
    print(f"Raw rows:     {raw_rows:,}")
    print(f"Clean rows:   {df.shape[0]:,}")
    print(f"Rows removed: {raw_rows - df.shape[0]:,} "
          f"({(raw_rows - df.shape[0])/raw_rows*100:.1f}%)")
    print(f"Columns:      {df.shape[1]}")

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    print("=" * 55)

    return df


if __name__ == "__main__":
    clean_dataset(
        input_path='data/MachineLearningRating_v3.txt',
        output_path='data/cleaned_insurance_data.csv'
    )