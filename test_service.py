from src.forecast_service import ForecastService


DATA_PATH = "data/synthetic_historical.csv"
SITE_CONFIG_PATH = "sample_sites.json"


def main():

    print("=" * 60)
    print("HACKOUT-2026 SERVICE TEST")
    print("=" * 60)

    print("\n[1] Initializing service...")

    service = ForecastService(
        DATA_PATH,
        SITE_CONFIG_PATH
    )

    service.initialize()

    print("Service initialized successfully.")

    print("\n[2] Available sites...")

    sites = service.get_sites()

    for site in sites:
        print(f"- {site}")

    print("\n[3] Generating forecast...")

    result = service.forecast(
        site_name="Western_Region_Solar_Park",
        hours=24,
        base_demand_mw=30.0
    )

    forecast = result["forecast"]
    decisions = result["decisions"]
    metrics = result["metrics"]

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

    print("\nGrid Decisions:")

    print(
        decisions[
            [
                "Timestamp",
                "Predicted_MW",
                "Status",
                "Alert"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nFeature Importance:")

    importance = (
        service.get_feature_importance()
    )

    print(
        importance.to_string(
            index=False
        )
    )
    print("\n[4] Testing What-If scenario...")
    
    scenario = service.compare_scenarios(
            site_name="Western_Region_Solar_Park",
            hours=24,
            base_demand_mw=30.0,
            weather_overrides={
                "AMBIENT_TEMPERATURE": 35.0,
                "MODULE_TEMPERATURE": 45.0,
                "IRRADIATION": 800.0
            }
        )
    
    comparison = scenario["comparison"]
    
    print(
            comparison.to_string(
                index=False
            )
        )
    
    print("\nScenario impact:")
    
    print(
            f"Average generation change: "
            f"{comparison['Impact_MW'].mean():.3f} MW"
        )
    print(
            f"Maximum generation change: "
            f"{comparison['Impact_MW'].max():.3f} MW"
        )
    
    print("\n" + "=" * 60)
    print("SERVICE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    

if __name__ == "__main__":
    main()