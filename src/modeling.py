"""
modeling.py
Reusable modeling functions for ACIS insurance
risk-based pricing system.

Models implemented:
- Linear Regression (baseline)
- Random Forest
- XGBoost

Tasks:
- Claim severity regression (TotalClaims where > 0)
- Claim probability classification (HasClaim 0/1)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score,
    recall_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# ─── Feature Engineering ────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.

    Features engineered:
    - VehicleAge: how old the vehicle is
    - HasAlarm: binary security feature
    - HasTracker: binary security feature
    - SecurityScore: combined security (0-2)
    - IsHighValue: vehicle value above R500k
    - IsCommercial: commercial vehicle flag

    Args:
        df: prepared DataFrame

    Returns:
        df: DataFrame with new feature columns added
    """
    df = df.copy()

    # Vehicle age — current year relative to dataset period
    # Dataset is from 2014-2015, use 2015 as reference
    if 'RegistrationYear' in df.columns:
        df['VehicleAge'] = 2015 - df['RegistrationYear']
        # Cap unrealistic ages
        df['VehicleAge'] = df['VehicleAge'].clip(0, 50)
        print("  ✅ VehicleAge engineered")

    # Security features
    if 'AlarmImmobiliser' in df.columns:
        df['HasAlarm'] = (df['AlarmImmobiliser'] == 'Yes').astype(int)
    if 'TrackingDevice' in df.columns:
        df['HasTracker'] = (df['TrackingDevice'] == 'Yes').astype(int)
    if 'HasAlarm' in df.columns and 'HasTracker' in df.columns:
        df['SecurityScore'] = df['HasAlarm'] + df['HasTracker']
        print("  ✅ SecurityScore engineered")

    # High value vehicle flag
    if 'CustomValueEstimate' in df.columns:
        df['IsHighValue'] = (
            df['CustomValueEstimate'] > 500000
        ).astype(int)
        print("  ✅ IsHighValue engineered")

    # Commercial vehicle flag
    if 'VehicleType' in df.columns:
        commercial_types = [
            'Medium Commercial',
            'Heavy Commercial',
            'Light Commercial'
        ]
        df['IsCommercial'] = (
            df['VehicleType'].isin(commercial_types)
        ).astype(int)
        print("  ✅ IsCommercial engineered")

    return df


# ─── Data Preparation ───────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list = None
) -> tuple:
    """
    Prepare feature matrix X and target vector y.

    Steps:
    1. Drop irrelevant ID/date columns
    2. Encode categorical columns with LabelEncoder
    3. Fill any remaining missing values
    4. Separate X and y

    Args:
        df         : full DataFrame
        target_col : name of target column
        drop_cols  : additional columns to drop

    Returns:
        X, y, feature_names, encoders
    """
    df = df.copy()

    # Always drop these — IDs and dates leak info
    # or have no predictive value
    always_drop = [
        'UnderwrittenCoverID', 'PolicyID',
        'TransactionMonth', 'VehicleIntroDate',
        'LossRatio',   # derived from target — data leakage
        'Margin',      # derived from target — data leakage
    ]

    # If predicting TotalClaims, also drop TotalPremium
    # to avoid leakage (they're both financial outcomes)
    if target_col == 'TotalClaims':
        always_drop.append('TotalPremium')
    if target_col == 'TotalPremium':
        always_drop.append('TotalClaims')

    if drop_cols:
        always_drop.extend(drop_cols)

    # Only drop columns that actually exist
    cols_to_drop = [c for c in always_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Also drop HasClaim if it exists and we're not targeting it
    if 'HasClaim' in X.columns and target_col != 'HasClaim':
        X = X.drop(columns=['HasClaim'])

    # Drop any datetime/timedelta columns that remain —
    # they cannot be encoded and cause NaN issues downstream
    datetime_cols = X.select_dtypes(
        include=['datetime64', 'datetimetz', 'timedelta64']
    ).columns.tolist()
    if datetime_cols:
        X = X.drop(columns=datetime_cols)
        print(f"  ℹ️  Dropped datetime columns: {datetime_cols}")

    # Encode categorical columns
    # LabelEncoder converts text to numbers
    # e.g. 'Gauteng' → 3, 'Western Cape' → 8
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        # Handle NaN by filling with 'Unknown' first
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Fill remaining numerical NaN with median
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Final safety net — catch any NaN that slipped through
    # (e.g. columns whose entire median is also NaN)
    if X.isnull().any().any():
        X = X.fillna(0)
        print("  ⚠️  Some NaN values filled with 0 (fallback)")

    feature_names = X.columns.tolist()
    print(f"  ✅ Features prepared: {len(feature_names)} features")
    print(f"  ✅ Target: {target_col} ({len(y):,} samples)")

    return X, y, feature_names, encoders


# ─── Model Training ─────────────────────────────────────────

def train_severity_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Train and evaluate three regression models for
    claim severity prediction.

    Models: Linear Regression, Random Forest, XGBoost
    Metrics: RMSE, R²

    Args:
        X_train, y_train: training data
        X_test, y_test:   test data

    Returns:
        dict of {model_name: {model, metrics, predictions}}
    """
    results = {}

    models = {
        'Linear Regression': LinearRegression(),

        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1     # use all CPU cores
        ),

        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    }

    print("\nTraining severity models...")
    print("=" * 55)

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict — clip negatives (claims can't be negative)
        y_pred = np.maximum(model.predict(X_test), 0)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model':       model,
            'predictions': y_pred,
            'rmse':        rmse,
            'r2':          r2,
        }

        print(f"    RMSE: R {rmse:,.2f}")
        print(f"    R²:   {r2:.4f}")

    return results


def train_probability_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Train and evaluate three classification models for
    claim probability prediction.

    Models: Logistic Regression, Random Forest, XGBoost
    Metrics: Accuracy, Precision, Recall, F1

    Args:
        X_train, y_train: training data
        X_test, y_test:   test data

    Returns:
        dict of {model_name: {model, metrics, predictions}}
    """
    results = {}

    # Class imbalance: ~0.28% of policies have claims
    # scale_pos_weight handles this in XGBoost
    # class_weight='balanced' handles it in sklearn
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count
    print(f"\n  Class balance — No claim: {neg_count:,} | "
          f"Claim: {pos_count:,} | Scale: {scale:.1f}x")

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),

        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale,  # handles class imbalance
            random_state=42,
            verbosity=0,
            eval_metric='logloss'
        )
    }

    print("\nTraining probability models...")
    print("=" * 55)

    for name, model in models.items():
        print(f"\n  Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            'model':         model,
            'predictions':   y_pred,
            'probabilities': y_prob,
            'accuracy':      acc,
            'precision':     prec,
            'recall':        rec,
            'f1':            f1,
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1 Score:  {f1:.4f}")

    return results


# ─── Comparison Table ────────────────────────────────────────

def compare_models(
    results: dict,
    task: str = 'regression'
) -> pd.DataFrame:
    """
    Build a comparison DataFrame for all models.

    Args:
        results: output from train_severity_models or
                 train_probability_models
        task:    'regression' or 'classification'

    Returns:
        DataFrame with model comparison metrics
    """
    rows = []
    for name, res in results.items():
        if task == 'regression':
            rows.append({
                'Model':    name,
                'RMSE':     f"R {res['rmse']:,.2f}",
                'R²':       f"{res['r2']:.4f}",
                'RMSE_raw': res['rmse'],
                'R2_raw':   res['r2'],
            })
        else:
            rows.append({
                'Model':     name,
                'Accuracy':  f"{res['accuracy']:.4f}",
                'Precision': f"{res['precision']:.4f}",
                'Recall':    f"{res['recall']:.4f}",
                'F1 Score':  f"{res['f1']:.4f}",
                'F1_raw':    res['f1'],
            })

    df = pd.DataFrame(rows)
    return df


# ─── Premium Calculator ──────────────────────────────────────

def calculate_premium(
    p_claim: float,
    predicted_severity: float,
    expense_loading: float = 0.15,
    profit_margin: float = 0.10
) -> dict:
    """
    Apply the risk-based pricing formula:

    Premium = (P(claim) × Predicted Severity)
            + Expense Loading
            + Profit Margin

    Args:
        p_claim            : probability of a claim (0-1)
        predicted_severity : predicted claim amount if claim occurs
        expense_loading    : fraction for operating expenses (default 15%)
        profit_margin      : target profit fraction (default 10%)

    Returns:
        dict with premium breakdown
    """
    expected_loss = p_claim * predicted_severity
    expense_amount = expected_loss * expense_loading
    profit_amount = expected_loss * profit_margin
    total_premium = expected_loss + expense_amount + profit_amount

    return {
        'p_claim':             round(p_claim, 4),
        'predicted_severity':  round(predicted_severity, 2),
        'expected_loss':       round(expected_loss, 2),
        'expense_loading':     round(expense_amount, 2),
        'profit_margin':       round(profit_amount, 2),
        'recommended_premium': round(total_premium, 2),
    }
