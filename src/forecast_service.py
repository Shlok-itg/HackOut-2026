from src.ml_engine import train_model
from src.decision_matrix import GridDecisionMatrix


class ForecastService:

    def __init__(
        self,
        data_path,
        site_config_path
    ):
        self.data_path = data_path
        self.site_config_path = site_config_path

        self.model = None
        self.df = None
        self.metrics = None
        self.matrix = None

    def initialize(self):

        (
            self.model,
            self.df,
            self.metrics
        ) = train_model(
            self.data_path
        )

        self.matrix = GridDecisionMatrix(
            self.site_config_path
        )

    def forecast(
        self,
        site_name,
        hours=24,
        base_demand_mw=30.0,
        weather_overrides=None
    ):

        if self.model is None:
            raise RuntimeError(
                "ForecastService has not been initialized."
            )

        if self.matrix is None:
            raise RuntimeError(
                "Decision matrix has not been initialized."
            )

        forecast = self.model.forecast_future(
            self.df,
            hours=hours,
            weather_overrides=weather_overrides
        )

        decisions = self.matrix.evaluate_forecast(
            site_name=site_name,
            forecast_df=forecast,
            base_demand_mw=base_demand_mw
        )

        decision_summary = (
            self.matrix.summarize_forecast(
                decisions
            )
        )

        return {
            "forecast": forecast,
            "decisions": decisions,
            "decision_summary": decision_summary,
            "metrics": self.metrics
        }

    def compare_scenarios(
        self,
        site_name,
        hours=24,
        base_demand_mw=30.0,
        weather_overrides=None
    ):

        if self.model is None:
            raise RuntimeError(
                "ForecastService has not been initialized."
            )

        if self.matrix is None:
            raise RuntimeError(
                "Decision matrix has not been initialized."
            )

        if weather_overrides is None:
            weather_overrides = {}

        baseline = self.model.forecast_future(
            self.df,
            hours=hours
        )

        what_if = self.model.forecast_future(
            self.df,
            hours=hours,
            weather_overrides=weather_overrides
        )

        baseline_decisions = (
            self.matrix.evaluate_forecast(
                site_name=site_name,
                forecast_df=baseline,
                base_demand_mw=base_demand_mw
            )
        )

        what_if_decisions = (
            self.matrix.evaluate_forecast(
                site_name=site_name,
                forecast_df=what_if,
                base_demand_mw=base_demand_mw
            )
        )

        comparison = what_if[
            [
                "Timestamp"
            ]
        ].copy()

        comparison["Baseline_MW"] = (
            baseline["Predicted_MW"].values
        )

        comparison["What_If_MW"] = (
            what_if["Predicted_MW"].values
        )

        comparison["Impact_MW"] = (
            comparison["What_If_MW"]
            - comparison["Baseline_MW"]
        )

        comparison["Baseline_Status"] = (
            baseline_decisions["Status"].values
        )

        comparison["What_If_Status"] = (
            what_if_decisions["Status"].values
        )

        return {
            "baseline": baseline,
            "baseline_decisions": baseline_decisions,
            "what_if": what_if,
            "what_if_decisions": what_if_decisions,
            "comparison": comparison
        }

    def get_sites(self):

        if self.matrix is None:
            raise RuntimeError(
                "ForecastService has not been initialized."
            )

        return self.matrix.get_available_sites()

    def get_feature_importance(self):

        if self.model is None:
            raise RuntimeError(
                "ForecastService has not been initialized."
            )

        return self.model.feature_importance()