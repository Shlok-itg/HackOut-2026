import streamlit as st
import plotly.graph_objects as go

from src.forecast_service import ForecastService


DATA_PATH = "data/synthetic_historical.csv"
SITE_CONFIG_PATH = "sample_sites.json"


st.set_page_config(
    page_title="Energy-Cast",
    page_icon="⚡",
    layout="wide"
)


@st.cache_resource
def load_service():

    service = ForecastService(
        DATA_PATH,
        SITE_CONFIG_PATH
    )

    service.initialize()

    return service


service = load_service()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("⚡ Energy-Cast")

st.subheader(
    "AI-Powered Renewable Energy Forecasting"
)

st.write(
    "Forecast renewable generation and evaluate "
    "grid stability for the next 24–72 hours."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Forecast Controls")

sites = [
    "Western_Region_Solar_Park"
]

selected_site = st.sidebar.selectbox(
    "Select Plant",
    sites
)

forecast_hours = st.sidebar.selectbox(
    "Forecast Horizon (24 Hrs, 48 Hrs, 72 Hrs)",
    [24, 48, 72],
    index=0
)


base_demand = st.sidebar.number_input(
    "Base Grid Demand (MW)",
    min_value=0.0,
    max_value=200.0,
    value=30.0,
    step=1.0
)


st.sidebar.subheader("⚡ What-If Scenario")
st.sidebar.caption(
    "Change weather conditions and "
    "observe the predicted generation impact."
)

ambient_temperature = st.sidebar.slider(
    "Ambient Temperature (°C)",
    min_value=10.0,
    max_value=45.0,
    value=30.0,
    step=0.5
)


module_temperature = st.sidebar.slider(
    "Module Temperature (°C)",
    min_value=10.0,
    max_value=70.0,
    value=35.0,
    step=0.5
)


irradiation = st.sidebar.slider(
    "Irradiation (W/m²)",
    min_value=0.0,
    max_value=1200.0,
    value=500.0,
    step=10.0
)
st.sidebar.caption(
    f"Current scenario: "
    f"{irradiation:.0f} W/m²"
)
if irradiation < 200:
    st.sidebar.caption(
        "Low sunlight scenario"
    )

elif irradiation < 600:
    st.sidebar.caption(
        "Moderate sunlight scenario"
    )

else:
    st.sidebar.caption(
        "High sunlight scenario"
    )

st.sidebar.divider()
#automatic plot update on slider change
run_forecast = st.sidebar.button(
    "⚡ Generate Forecast",
    #use_container_width=True
)
run_forecast=True


# ============================================================
# FORECAST EXECUTION
# ============================================================

if run_forecast:

    with st.spinner(
        "Generating renewable energy forecast..."
    ):

        scenario = service.compare_scenarios(
            site_name=selected_site,
            hours=forecast_hours,
            base_demand_mw=base_demand,
            weather_overrides={
                "AMBIENT_TEMPERATURE":
                    ambient_temperature,

                "MODULE_TEMPERATURE":
                    module_temperature,

                "IRRADIATION":
                    irradiation
            }
        )


    baseline = scenario["baseline"]

    baseline_decisions = scenario[
        "baseline_decisions"
    ]

    what_if = scenario["what_if"]

    what_if_decisions = scenario[
        "what_if_decisions"
    ]

    comparison = scenario[
        "comparison"
    ]


    # ========================================================
    # SITE CONFIGURATION
    # ========================================================

    site_config = service.matrix.get_site_config(
        selected_site
    )


    max_capacity = float(
        site_config["max_capacity_mw"]
    )


    curtailment_threshold = float(
        site_config[
            "curtailment_warning_threshold"
        ]
    )


    curtailment_threshold_mw = (
        max_capacity
        * curtailment_threshold
    )


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        "Forecast generated successfully."
    )
    st.info(
    f"Scenario: {selected_site} | "
    f"{forecast_hours}-hour forecast | "
    f"Irradiation: {irradiation:.0f} W/m²"
)


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    metrics = service.metrics


    st.subheader(
        "Model Performance"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "R²",
            f"{metrics['R2']:.4f}"
        )


    with col2:

        st.metric(
            "MAE",
            f"{metrics['MAE']:.3f} MW"
        )


    with col3:

        st.metric(
            "RMSE",
            f"{metrics['RMSE']:.3f} MW"
        )


    # ========================================================
    # FORECAST SUMMARY
    # ========================================================

    st.subheader(
        "Forecast Summary"
    )


    daylight_forecast = what_if[
        what_if["IRRADIATION"] > 0
    ]


    if len(daylight_forecast) > 0:

        minimum_daylight = (
            daylight_forecast[
                "Predicted_MW"
            ].min()
        )

    else:

        minimum_daylight = 0.0


    maximum_forecast = (
        what_if["Predicted_MW"].max()
    )


    average_forecast = (
        what_if["Predicted_MW"].mean()
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Maximum Forecast",
            f"{maximum_forecast:.2f} MW"
        )


    with col2:

        st.metric(
            "Minimum Daylight Forecast",
            f"{minimum_daylight:.2f} MW"
        )


    with col3:

        st.metric(
            "Average Forecast",
            f"{average_forecast:.2f} MW"
        )


    # ========================================================
    # CAPACITY UTILIZATION
    # ========================================================

    st.subheader(
        "Plant Capacity Utilization"
    )
    peak_utilization = float(
    maximum_forecast / max_capacity
)

    progress_value = min(
    float(peak_utilization),
    1.0
)

    st.progress(
    progress_value
)


    st.caption(
    f"Peak forecast utilization: "
    f"{float(peak_utilization) * 100:.1f}% "
    f"of {float(max_capacity):.1f} MW capacity")


    if maximum_forecast >= max_capacity:

        st.error(
            "⚠️ Forecast exceeds plant capacity. "
            "Immediate curtailment or storage routing "
            "is required."
        )


    elif maximum_forecast >= curtailment_threshold_mw:

        st.warning(
            "⚠️ Forecast is inside the curtailment "
            "warning zone. Prepare storage or "
            "curtailment routing."
        )


    else:

        st.success(
            "Forecast remains below the curtailment "
            "warning threshold."
        )


    # ========================================================
    # WHAT-IF IMPACT
    # ========================================================

    st.subheader(
        "What-If Impact"
    )


    average_baseline = (
        comparison["Baseline_MW"].mean()
    )


    average_what_if = (
        comparison["What_If_MW"].mean()
    )


    average_impact = (
        comparison["Impact_MW"].mean()
    )


    maximum_impact = (
        comparison["Impact_MW"].max()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Baseline Average",
            f"{average_baseline:.2f} MW"
        )


    with col2:

        st.metric(
            "What-If Average",
            f"{average_what_if:.2f} MW"
        )


    with col3:

        st.metric(
            "Average Impact",
            f"{average_impact:+.2f} MW"
        )


    with col4:

        st.metric(
            "Maximum Impact",
            f"{maximum_impact:+.2f} MW"
        )


    # ========================================================
    # BASELINE VS WHAT-IF CHART
    # ========================================================

    st.subheader(
        "Baseline vs What-If Forecast"
    )


    fig = go.Figure()


    fig.add_trace(
    go.Scatter(
        x=baseline["Timestamp"],
        y=baseline["Predicted_MW"],
        mode="lines+markers",
        name="Baseline Forecast",
        line=dict(
            color="#002FFF",
            width=3
        ),
        marker=dict(
            size=6
        )
    )
)


    fig.add_trace(
    go.Scatter(
        x=what_if["Timestamp"],
        y=what_if["Predicted_MW"],
        mode="lines+markers",
        name="What-If Forecast",
        line=dict(
            color="#FFB000",
            width=3
        ),
        marker=dict(
            size=6
        )
    )
)


    fig.add_trace(
    go.Scatter(
        x=baseline["Timestamp"],
        y=baseline["Upper_MW"],
        mode="lines",
        name="Upper Confidence",
        line=dict(
            color="rgba(0, 229, 255, 0.6)",
            width=1,
            dash="dash"
        )
    )
)

    fig.add_trace(
    go.Scatter(
        x=baseline["Timestamp"],
        y=baseline["Lower_MW"],
        mode="lines",
        name="Lower Confidence",
        line=dict(
            color="rgba(0, 229, 255, 0.6)",
            width=1,
            dash="dash"
        )
    )
)


    fig.add_hline(
        y=max_capacity,
        line_dash="dash",
        annotation_text=(
            f"Max Capacity: "
            f"{max_capacity:.1f} MW"
        ),
        annotation_position="top right"
    )


    fig.add_hline(
        y=curtailment_threshold_mw,
        line_dash="dot",
        annotation_text=(
            f"Curtailment Threshold: "
            f"{curtailment_threshold_mw:.1f} MW"
        ),
        annotation_position="bottom right"
    )


    fig.add_hline(
        y=base_demand,
        line_dash="dashdot",
        annotation_text=(
            f"Base Demand: "
            f"{base_demand:.1f} MW"
        ),
        annotation_position="bottom left"
    )


    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Generation (MW)",
        hovermode="x unified",
        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # WHAT-IF GRID STABILITY
    # ========================================================

    st.subheader(
        "What-If Grid Stability"
    )


    what_if_summary = (
        service.matrix.summarize_forecast(
            what_if_decisions
        )
    )


    priority = what_if_summary[
        "priority"
    ]


    if priority == "GREEN":

        st.success(
            "GREEN — What-If scenario remains "
            "within stable operating conditions."
        )


    elif priority == "AMBER":

        st.warning(
            "AMBER — What-If scenario creates "
            "a generation shortfall."
        )


    else:

        st.error(
            "RED — What-If scenario creates "
            "a critical generation condition."
        )


    st.write(
        "Recommended action: "
        + what_if_summary["recommendation"]
    )


    # ========================================================
    # GRID STATUS COUNTERS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Stable Hours",
            what_if_summary["green_hours"]
        )


    with col2:

        st.metric(
            "Warning Hours",
            what_if_summary["amber_hours"]
        )


    with col3:

        st.metric(
            "Critical Hours",
            what_if_summary["red_hours"]
        )


    # ========================================================
    # GRID STATUS TIMELINE
    # ========================================================

    st.subheader(
        "Grid Status Timeline"
    )


    status_values = (
        what_if_decisions["Status"]
        .map({
            "GREEN": 0,
            "AMBER": 1,
            "RED": 2
        })
    )


    status_fig = go.Figure()


    status_fig.add_trace(
        go.Scatter(
            x=what_if_decisions["Timestamp"],
            y=status_values,
            mode="lines+markers",
            name="Grid Status",
            text=what_if_decisions["Status"],
            hovertemplate=(
                "%{x}<br>"
                "Status: %{text}<extra></extra>"
            )
        )
    )


    status_fig.update_layout(
        xaxis_title="Time",
        yaxis=dict(
            title="Grid Status",
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=[
                "GREEN",
                "AMBER",
                "RED"
            ],
            range=[-0.2, 2.2]
        ),
        height=300,
        hovermode="x"
    )


    st.plotly_chart(
        status_fig,
        use_container_width=True
    )


    # ========================================================
    # HOURLY WHAT-IF DECISIONS
    # ========================================================

    st.subheader(
        "Hourly What-If Decisions"
    )


    st.dataframe(
        what_if_decisions[
            [
                "Timestamp",
                "Predicted_MW",
                "Status",
                "Alert",
                "Action"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "Model Feature Importance"
    )


    importance = (
        service.get_feature_importance()
    )

    importance_fig = go.Figure()
    importance_fig.add_trace(
    go.Bar(
        x=importance["Importance"],
        y=importance["Feature"],
        orientation="h"
    )
    )
    importance_fig.update_layout(
    title="XGBoost Feature Importance",
    xaxis_title="Importance",
    yaxis_title="Feature",
    height=450
    )
    st.plotly_chart(
    importance_fig,
    use_container_width=True
    )
    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )
    st.subheader(
    "Executive Decision Summary"
    )
    if priority == "RED":
        st.error(
        f"RED ALERT — {what_if_summary['red_hours']} "
        f"forecast hour(s) require immediate grid action."
    )
    elif priority == "AMBER":
        st.warning(
        f"AMBER ALERT — {what_if_summary['amber_hours']} "
        f"forecast hour(s) show a generation shortfall."
    )
    else:
        st.success(
        "GREEN — No critical grid condition "
        "is detected in the forecast."
    )
        st.write(what_if_summary["recommendation"]
)

else:

    st.info(
        "Select the plant, forecast horizon, "
        "grid demand, and What-If weather values, "
        "then click 'Generate Forecast'."
    )