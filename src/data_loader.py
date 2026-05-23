
import pandas as pd
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the insurance dataset from a CSV or pipe-delimited text file.

    Args:
        filepath: path to the data file

    Returns:
        df: raw DataFrame
    """
    # Check the file exists before trying to load it
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    # Detect file extension
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.txt':
        # The ACIS dataset uses | as a separator
        df = pd.read_csv(filepath, sep='|', low_memory=False)
    elif ext == '.csv':
        df = pd.read_csv(filepath, low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"✅ Data loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix column data types:
    - Convert TransactionMonth to datetime
    - Ensure numerical columns are numeric

    Args:
        df: raw DataFrame

    Returns:
        df: DataFrame with corrected types
    """
    df = df.copy()

    # Convert date column
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(
            df['TransactionMonth'], errors='coerce'
        )
        print("✅ TransactionMonth converted to datetime")

    # Columns that must be numeric
    numeric_cols = [
        'TotalPremium', 'TotalClaims', 'SumInsured',
        'CalculatedPremiumPerTerm', 'CustomValueEstimate',
        'Cylinders', 'Kilowatts', 'NumberOfDoors'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("✅ Numeric columns fixed")
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the two core business metrics:
    - LossRatio = TotalClaims / TotalPremium
    - Margin    = TotalPremium - TotalClaims

    Args:
        df: DataFrame with TotalPremium and TotalClaims

    Returns:
        df: DataFrame with new columns added
    """
    df = df.copy()

    # Avoid division by zero — replace 0 premiums with NaN
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, float('nan'))
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']

    print("✅ Derived metrics added: LossRatio, Margin")
    return df
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize known inconsistent column names
    so the rest of the code can use consistent references.

    The raw data has mixed casing:
    'make', 'kilowatts', 'cubiccapacity', 'bodytype', 'mmcode'
    We rename these to Title Case to match documentation.

    Args:
        df: raw DataFrame

    Returns:
        df: DataFrame with renamed columns
    """
    df = df.copy()

    # Map from actual column name → what we want to call it
    rename_map = {
        'make'          : 'Make',
        'kilowatts'     : 'Kilowatts',
        'cubiccapacity' : 'Cubiccapacity',
        'bodytype'      : 'Bodytype',
        'mmcode'        : 'Mmcode',
    }

    # Only rename columns that actually exist
    # This prevents errors if the data changes
    rename_map = {
        k: v for k, v in rename_map.items()
        if k in df.columns
    }

    df = df.rename(columns=rename_map)
    print(f"✅ Columns standardized: {list(rename_map.keys())}")
    return df

def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Master function: load + fix types + add metrics in one call.
    This is what your notebooks will call.

    Args:
        filepath: path to the data file

    Returns:
        df: fully prepared DataFrame
    """
    df = load_data(filepath)
    df = fix_dtypes(df)
    df = standardize_columns(df)
    df = add_derived_metrics(df)
    return df