import json
import pandas as pd

class GridDecisionMatrix:

    def __init__(self, site_config_path):
        self.site_config = self._load_site_config(
            site_config_path
        )
    def get_available_sites(self):
        return list(
            self.site_config.keys()
        )
    def _load_site_config(self, path):
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def get_site_config(self, site_name):
        if site_name not in self.site_config:
            raise ValueError(
                f"Unknown site: {site_name}"
            )

        return self.site_config[site_name]

    def evaluate(
        self,
        site_name,
        predicted_power_mw,
        base_demand_mw
    ):
        config = self.get_site_config(
            site_name
        )

        max_capacity = float(
            config["max_capacity_mw"]
        )

        curtailment_threshold = float(
            config["curtailment_warning_threshold"]
        )

        predicted_power_mw = float(
            predicted_power_mw
        )

        base_demand_mw = float(
            base_demand_mw
        )

        threshold_mw = (
            max_capacity
            * curtailment_threshold
        )

        utilization = 0.0

        if max_capacity > 0:
            utilization = (
                predicted_power_mw
                / max_capacity
            )

        if predicted_power_mw >= max_capacity:
            return {
                "status": "RED",
                "alert": "Critical over-generation",
                "action": (
                    "Curtail generation or route "
                    "excess power to battery storage."
                ),
                "predicted_power_mw": predicted_power_mw,
                "base_demand_mw": base_demand_mw,
                "capacity_mw": max_capacity,
                "utilization_percent": utilization * 100,
            }

        if predicted_power_mw >= threshold_mw:
            return {
                "status": "RED",
                "alert": "Curtailment warning",
                "action": (
                    "Prepare battery storage or "
                    "curtailment route."
                ),
                "predicted_power_mw": predicted_power_mw,
                "base_demand_mw": base_demand_mw,
                "capacity_mw": max_capacity,
                "utilization_percent": utilization * 100,
            }

        if predicted_power_mw < base_demand_mw:
            deficit = (
                base_demand_mw
                - predicted_power_mw
            )

            return {
                "status": "AMBER",
                "alert": "Generation shortfall",
                "action": (
                    "Dispatch backup or peaker "
                    "generation."
                ),
                "predicted_power_mw": predicted_power_mw,
                "base_demand_mw": base_demand_mw,
                "deficit_mw": deficit,
                "capacity_mw": max_capacity,
                "utilization_percent": utilization * 100,
            }

        return {
            "status": "GREEN",
            "alert": "Grid conditions stable",
            "action": (
                "Continue normal generation "
                "and monitoring."
            ),
            "predicted_power_mw": predicted_power_mw,
            "base_demand_mw": base_demand_mw,
            "capacity_mw": max_capacity,
            "utilization_percent": utilization * 100,
        }

    def evaluate_forecast(
        self,
        site_name,
        forecast_df,
        base_demand_mw
    ):
        results = []

        for _, row in forecast_df.iterrows():

            decision = self.evaluate(
                site_name=site_name,
                predicted_power_mw=row["Predicted_MW"],
                base_demand_mw=base_demand_mw
            )

            results.append({
                "Timestamp": row["Timestamp"],
                "Predicted_MW": row["Predicted_MW"],
                "Lower_MW": row["Lower_MW"],
                "Upper_MW": row["Upper_MW"],
                "Status": decision["status"],
                "Alert": decision["alert"],
                "Action": decision["action"],
                "Capacity_MW": decision["capacity_mw"],
                "Utilization_Percent": (
                    decision["utilization_percent"]
                )
            })

        return pd.DataFrame(results)
    def summarize_forecast(self, decisions):

        status_counts = (
            decisions["Status"]
            .value_counts()
            .to_dict()
        )

        red_hours = decisions[
            decisions["Status"] == "RED"
        ]

        amber_hours = decisions[
            decisions["Status"] == "AMBER"
        ]

        if len(red_hours) > 0:

            priority = "RED"

            critical_row = red_hours.iloc[0]

            recommendation = critical_row["Action"]

        elif len(amber_hours) > 0:

            priority = "AMBER"

            critical_row = amber_hours.iloc[0]

            recommendation = critical_row["Action"]

        else:

            priority = "GREEN"

            recommendation = (
                "Continue normal generation "
                "and monitoring."
            )

        return {
            "priority": priority,
            "green_hours": status_counts.get(
                "GREEN", 0
            ),
            "amber_hours": status_counts.get(
                "AMBER", 0
            ),
            "red_hours": status_counts.get(
                "RED", 0
            ),
            "recommendation": recommendation
        }