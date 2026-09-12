from src.ml_engine import train_model
from src.decision_matrix import GridDecisionMatrix


DATA_PATH = "data/synthetic_historical.csv"
SITE_CONFIG_PATH = "sample_sites.json"

SITE_NAME = "Western_Region_Solar_Park"
BASE_DEMAND_MW = 30.0
FORECAST_HOURS = 24


def main():

    print("=" * 60)
    print("HACKOUT-2026 BACKEND TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. TRAIN MODEL
    # ---------------------------------------------------------

    print("\n[1] Loading and training model...")

    model, df, metrics = train_model(
        DATA_PATH
    )

    print("Model trained successfully.")

    print("\nDataset:")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    period = model.get_training_period(df)

    print("\nTraining Period:")
    print(
        f"{period['train_start']} "
        f"→ "
        f"{period['train_end']}"
    )

    print("\nTesting Period:")
    print(
        f"{period['test_start']} "
        f"→ "
        f"{period['test_end']}"
    )

    print("\nModel Metrics:")
    print(
        f"R²   : {metrics['R2']:.4f}"
    )
    print(
        f"MAE  : {metrics['MAE']:.4f} MW"
    )
    print(
        f"RMSE : {metrics['RMSE']:.4f} MW"
    )

    # ---------------------------------------------------------
    # 2. SINGLE PREDICTION TEST
    # ---------------------------------------------------------

    print("\n[2] Testing single prediction...")

    latest = df.iloc[-1]

    features = {
        "AMBIENT_TEMPERATURE":
            latest["AMBIENT_TEMPERATURE"],

        "MODULE_TEMPERATURE":
            latest["MODULE_TEMPERATURE"],

        "IRRADIATION":
            latest["IRRADIATION"],

        "Hour":
            latest["Hour"],

        "DayOfWeek":
            latest["DayOfWeek"],

        "Month":
            latest["Month"],

        "Hour_sin":
            latest["Hour_sin"],

        "Hour_cos":
            latest["Hour_cos"],

        "Month_sin":
            latest["Month_sin"],

        "Month_cos":
            latest["Month_cos"]
    }

    prediction_result = (
        model.predict_with_confidence(
            features
        )
    )

    prediction = float(
        prediction_result["prediction"][0]
    )

    lower = float(
        prediction_result["lower_bound"][0]
    )

    upper = float(
        prediction_result["upper_bound"][0]
    )

    print(
        f"Predicted power : {prediction:.3f} MW"
    )

    print(
        f"Lower bound     : {lower:.3f} MW"
    )

    print(
        f"Upper bound     : {upper:.3f} MW"
    )

    # ---------------------------------------------------------
    # 3. FUTURE FORECAST
    # ---------------------------------------------------------

    print(
        f"\n[3] Generating "
        f"{FORECAST_HOURS}-hour forecast..."
    )

    forecast = model.forecast_future(
        df,
        hours=FORECAST_HOURS
    )

    print("\nForecast:")

    print(
        forecast[
            [
                "Timestamp",
                "Predicted_MW",
                "Lower_MW",
                "Upper_MW"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nForecast Summary:")

    print(
        f"Maximum predicted power : "
        f"{forecast['Predicted_MW'].max():.3f} MW"
    )

    print(
        f"Minimum predicted power : "
        f"{forecast['Predicted_MW'].min():.3f} MW"
    )

    print(
        f"Average predicted power : "
        f"{forecast['Predicted_MW'].mean():.3f} MW"
    )

    # ---------------------------------------------------------
    # 4. GRID DECISION MATRIX
    # ---------------------------------------------------------

    print(
        "\n[4] Evaluating forecast "
        "against grid limits..."
    )

    matrix = GridDecisionMatrix(
        SITE_CONFIG_PATH
    )

    decisions = matrix.evaluate_forecast(
        site_name=SITE_NAME,
        forecast_df=forecast,
        base_demand_mw=BASE_DEMAND_MW
    )

    print("\nGrid Decisions:")

    print(
        decisions[
            [
                "Timestamp",
                "Predicted_MW",
                "Status",
                "Alert",
                "Action"
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 5. DECISION SUMMARY
    # ---------------------------------------------------------

    print("\nDecision Summary:")

    status_counts = (
        decisions["Status"]
        .value_counts()
    )

    for status in [
        "GREEN",
        "AMBER",
        "RED"
    ]:

        count = status_counts.get(
            status,
            0
        )

        print(
            f"{status}: {count} hours"
        )

    # ---------------------------------------------------------
    # 6. FEATURE IMPORTANCE
    # ---------------------------------------------------------

    print("\n[5] Feature importance...")

    importance = (
        model.feature_importance()
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 7. AVAILABLE SITES
    # ---------------------------------------------------------

    print("\n[6] Available sites...")

    sites = (
        matrix.get_available_sites()
    )

    for site in sites:
        print(
            f"- {site}"
        )

    # ---------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("BACKEND TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()