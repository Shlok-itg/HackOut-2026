import pandas as pd
import numpy as np


TIME_COLUMN = "Timestamp"
TARGET_COLUMN = "AC_POWER_MW"


def load_dataset(path):
    df = pd.read_csv(path)

    if TIME_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{TIME_COLUMN}' not found."
        )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="coerce"
    )

    df = df.dropna(subset=[TIME_COLUMN])

    return df


def clean_raw_data(df):
    df = df.copy()

    df = df.drop_duplicates()

    numeric_columns = [
        "DC_POWER",
        "AC_POWER",
        "DAILY_YIELD",
        "TOTAL_YIELD",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.sort_values(TIME_COLUMN)

    return df


def aggregate_to_plant_level(df):
    required_columns = [
        TIME_COLUMN,
        "AC_POWER",
        "DC_POWER",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    grouped = (
        df.groupby(TIME_COLUMN)
        .agg({
            "AC_POWER": "sum",
            "DC_POWER": "sum",
            "AMBIENT_TEMPERATURE": "mean",
            "MODULE_TEMPERATURE": "mean",
            "IRRADIATION": "mean"
        })
        .reset_index()
    )

    grouped["AC_POWER_MW"] = (
        grouped["AC_POWER"] / 1000.0
    )

    grouped["DC_POWER_MW"] = (
        grouped["DC_POWER"] / 1000.0
    )

    return grouped


def add_time_features(df):
    df = df.copy()

    timestamp = df[TIME_COLUMN]

    df["Hour"] = timestamp.dt.hour
    df["Day"] = timestamp.dt.day
    df["DayOfWeek"] = timestamp.dt.dayofweek
    df["Month"] = timestamp.dt.month

    df["Hour_sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    df["Month_sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    return df


def interpolate_missing_values(df):
    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    return df


def preprocess_dataset(path):
    df = load_dataset(path)

    df = clean_raw_data(df)

    df = aggregate_to_plant_level(df)

    df = interpolate_missing_values(df)

    df = add_time_features(df)

    return df


def get_dataset_summary(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "start_time": df[TIME_COLUMN].min(),
        "end_time": df[TIME_COLUMN].max(),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "max_generation_mw": float(
            df[TARGET_COLUMN].max()
        ),
        "mean_generation_mw": float(
            df[TARGET_COLUMN].mean()
        )
    }