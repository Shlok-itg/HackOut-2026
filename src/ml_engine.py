import numpy as np
import pandas as pd

from xgboost import XGBRegressor


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_pipeline import preprocess_dataset


TARGET_COLUMN = "AC_POWER_MW"

FEATURE_COLUMNS = ["AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION","Hour","DayOfWeek","Month","Hour_sin","Hour_cos","Month_sin","Month_cos"]


class RenewableForecastModel:

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        self.is_trained = False

        self.metrics = {}

        self.residual_std = None

    def prepare_data(self, df):

        missing_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing feature columns: {missing_features}"
            )

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Missing target column: {TARGET_COLUMN}"
            )

        data = df[
            FEATURE_COLUMNS + [TARGET_COLUMN]
        ].copy()

        data = data.dropna()

        X = data[FEATURE_COLUMNS]
        y = data[TARGET_COLUMN]

        return X, y

    def train(self, df):

        X, y = self.prepare_data(df)

        split_index = int(len(X) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(
            X_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        residuals = (
            y_test.values - predictions
        )

        self.residual_std = float(
            np.std(residuals)
        )

        self.metrics = {
            "R2": float(r2),
            "MAE": float(mae),
            "RMSE": float(rmse)
        }

        self.is_trained = True

        return self.metrics

    def get_training_period(self, df):

        if "Timestamp" not in df.columns:
            raise ValueError(
                "Timestamp column not found."
            )

        split_index = int(len(df) * 0.8)

        train_start = df["Timestamp"].iloc[0]
        train_end = df["Timestamp"].iloc[
            split_index - 1
        ]

        test_start = df["Timestamp"].iloc[
            split_index
        ]
        test_end = df["Timestamp"].iloc[-1]

        return {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end
        }
    def predict(self, features):

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        if isinstance(features, dict):
            features = pd.DataFrame(
                [features]
            )

        features = features[
            FEATURE_COLUMNS
        ]

        prediction = self.model.predict(
            features
        )

        return np.maximum(
            prediction,
            0
        )

    def predict_with_confidence(
        self,
        features,
        confidence_multiplier=1.96
    ):

        predictions = self.predict(
            features
        )

        if self.residual_std is None:
            margin = 0.0
        else:
            margin = (
                confidence_multiplier
                * self.residual_std
            )

        lower = np.maximum(
            predictions - margin,
            0
        )

        upper = predictions + margin

        return {
            "prediction": predictions,
            "lower_bound": lower,
            "upper_bound": upper
        }
    def forecast_future(
        self,
        df,
        hours=72,
        weather_overrides=None
    ):
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        if "Timestamp" not in df.columns:
            raise ValueError(
                "Timestamp column not found."
            )

        if hours not in (24, 48, 72):
            raise ValueError(
                "Forecast horizon must be 24, 48, or 72 hours."
            )

        weather_columns = [
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]

        for column in weather_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Missing weather column: {column}"
                )

        # ----------------------------------------------------
        # Prepare hourly historical weather
        # ----------------------------------------------------

        history = df.copy()

        history["Timestamp"] = pd.to_datetime(
            history["Timestamp"]
        )

        history = history.sort_values(
            "Timestamp"
        )

        hourly_history = (
            history
            .set_index("Timestamp")[weather_columns]
            .resample("1h")
            .mean()
        )

        hourly_history = (
            hourly_history
            .interpolate(
                method="linear",
                limit_direction="both"
            )
        )

        # ----------------------------------------------------
        # Find complete historical days with useful
        # solar activity.
        # ----------------------------------------------------

        daily_irradiation = (
            hourly_history["IRRADIATION"]
            .resample("1D")
            .sum()
        )

        valid_days = daily_irradiation[
            daily_irradiation > 0
        ].index

        if len(valid_days) == 0:
            raise ValueError(
                "No historical daylight data was found."
            )

        # Use the most recent available daylight days.
        # Number of days depends on forecast horizon.
        days_needed = hours // 24

        selected_days = list(
            valid_days[-days_needed:]
        )

        # If fewer days are available, repeat the
        # latest valid day only as a fallback.
        while len(selected_days) < days_needed:
            selected_days.insert(
                0,
                selected_days[-1]
            )

        # ----------------------------------------------------
        # Build future timestamps
        # ----------------------------------------------------

        last_timestamp = (
            hourly_history.index.max()
        )

        last_timestamp = (
            last_timestamp.floor("h")
        )

        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=hours,
            freq="1h"
        )

        future = pd.DataFrame({
            "Timestamp": future_timestamps
        })

        # ----------------------------------------------------
        # Create day-by-day weather forecast
        # ----------------------------------------------------

        weather_blocks = []

        for day in selected_days:

            day_start = pd.Timestamp(day)

            day_end = (
                day_start
                + pd.Timedelta(days=1)
            )

            day_data = hourly_history.loc[
                (hourly_history.index >= day_start)
                &
                (hourly_history.index < day_end)
            ].copy()

            day_data = (
                day_data
                .reindex(
                    pd.date_range(
                        start=day_start,
                        periods=24,
                        freq="1h"
                    )
                )
            )

            day_data = (
                day_data
                .interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )

            weather_blocks.append(
                day_data.reset_index(drop=True)
            )

        recent_weather = pd.concat(
            weather_blocks,
            ignore_index=True
        )

        recent_weather = recent_weather.iloc[
            :hours
        ].copy()

        # Safety fallback
        if len(recent_weather) < hours:

            repeats = int(
                np.ceil(
                    hours
                    / len(recent_weather)
                )
            )

            recent_weather = pd.concat(
                [recent_weather] * repeats,
                ignore_index=True
            ).iloc[:hours]

        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------

        future["Hour"] = (
            future["Timestamp"].dt.hour
        )

        future["DayOfWeek"] = (
            future["Timestamp"].dt.dayofweek
        )

        future["Month"] = (
            future["Timestamp"].dt.month
        )

        future["Hour_sin"] = np.sin(
            2 * np.pi * future["Hour"] / 24
        )

        future["Hour_cos"] = np.cos(
            2 * np.pi * future["Hour"] / 24
        )

        future["Month_sin"] = np.sin(
            2 * np.pi * future["Month"] / 12
        )

        future["Month_cos"] = np.cos(
            2 * np.pi * future["Month"] / 12
        )

        # ----------------------------------------------------
        # Historical weather sequence
        # ----------------------------------------------------

        for column in weather_columns:

            future[column] = (
                recent_weather[column]
                .to_numpy()
            )

        # ----------------------------------------------------
        # What-If overrides
        # ----------------------------------------------------

        if weather_overrides:

            if "AMBIENT_TEMPERATURE" in weather_overrides:

                future["AMBIENT_TEMPERATURE"] = float(
                    weather_overrides[
                        "AMBIENT_TEMPERATURE"
                    ]
                )

            if "MODULE_TEMPERATURE" in weather_overrides:

                future["MODULE_TEMPERATURE"] = float(
                    weather_overrides[
                        "MODULE_TEMPERATURE"
                    ]
                )

            if "IRRADIATION" in weather_overrides:

                target_irradiation = (float(weather_overrides["IRRADIATION"])/1000.0
)

                daylight_mask = (future["IRRADIATION"] > 0.001)

                # Fallback daylight definition
                if not daylight_mask.any():

                    daylight_mask = (
                        (future["Hour"] >= 6)
                        &
                        (future["Hour"] <= 18)
                    )

                # Store original daylight profile
                original_daylight = (
                    future.loc[
                        daylight_mask,
                        "IRRADIATION"
                    ].copy()
                )

                if (
                    len(original_daylight) > 0
                    and original_daylight.mean() > 0
                ):

                    scale = (
                        target_irradiation
                        / original_daylight.mean()
                    )

                    future.loc[
                        daylight_mask,
                        "IRRADIATION"
                    ] = (
                        original_daylight
                        * scale
                    )

                else:

                    future.loc[
                        daylight_mask,
                        "IRRADIATION"
                    ] = target_irradiation

                # Night must remain zero
                future.loc[
                    ~daylight_mask,
                    "IRRADIATION"
                ] = 0.0

        # ----------------------------------------------------
        # Physical bounds
        # ----------------------------------------------------

        future["IRRADIATION"] = (
            future["IRRADIATION"]
            .clip(
                lower=0,
                upper=1.2
            )
        )

        future["AMBIENT_TEMPERATURE"] = (
            future["AMBIENT_TEMPERATURE"]
            .clip(
                lower=0,
                upper=60
            )
        )

        future["MODULE_TEMPERATURE"] = (
            future["MODULE_TEMPERATURE"]
            .clip(
                lower=0,
                upper=90
            )
        )

        # ----------------------------------------------------
        # XGBoost prediction
        # ----------------------------------------------------

        features = future[
            FEATURE_COLUMNS
        ]

        forecast = self.predict_with_confidence(
            features
        )

        future["Predicted_MW"] = (
            forecast["prediction"]
        )

        future["Lower_MW"] = (
            forecast["lower_bound"]
        )

        future["Upper_MW"] = (
            forecast["upper_bound"]
        )

        # ----------------------------------------------------
        # Solar night-time constraint
        # ----------------------------------------------------

        night_mask = (
            future["IRRADIATION"] <= 0.001
        )

        future.loc[
            night_mask,
            "Predicted_MW"
        ] = 0.0

        future.loc[
            night_mask,
            "Lower_MW"
        ] = 0.0

        future.loc[
            night_mask,
            "Upper_MW"
        ] = 0.0

        # ----------------------------------------------------
        # Forecast hour index
        # ----------------------------------------------------

        future["Forecast_Hour"] = (
            np.arange(1, hours + 1)
        )

        return future[
            [
                "Timestamp",
                "Forecast_Hour",
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION",
                "Predicted_MW",
                "Lower_MW",
                "Upper_MW"
            ]
        ]

    def feature_importance(self):

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        importance = (
            self.model.feature_importances_
        )

        result = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": importance
        })

        return result.sort_values(
            "Importance",
            ascending=False
        ).reset_index(drop=True)


def train_model(data_path):

    df = preprocess_dataset(
        data_path
    )

    forecast_model = RenewableForecastModel()

    metrics = forecast_model.train(df)

    return forecast_model, df, metrics