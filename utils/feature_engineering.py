import pandas as pd
import numpy as np

def create_features(df):
    """
    Create time-based and lag features for sales forecasting.

    Parameters:
    df (pd.DataFrame): Input dataframe with at least 'Date', 'Sales', 'Store'

    Returns:
    pd.DataFrame: DataFrame with added features
    """
    
    # Time-based calendar features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype("int16")
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype("int8")

    # Lag features (store-specific)
    df["SalesLag1"] = df.groupby("Store")["Sales"].shift(1)
    df["SalesLag7"] = df.groupby("Store")["Sales"].shift(7)

    # Moving average (trend smoothening)
    df["SalesMA7"] = df.groupby("Store")["Sales"].shift(1).rolling(7).mean()

    # Promo2SinceDate for duration calculation
    df["Promo2Since"] = pd.to_datetime(
        df["Promo2SinceYear"].fillna(0).astype(int).astype(str) + "-01",
        errors='coerce'
    )

    # (Optional) Days since promo2 started
    df["DaysSincePromo2"] = (df["Date"] - df["Promo2Since"]).dt.days
    df["DaysSincePromo2"] = df["DaysSincePromo2"].fillna(0).clip(lower=0)

    return df
