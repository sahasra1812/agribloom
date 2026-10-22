from pathlib import Path
from datetime import date, timedelta
import html
import json

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.optimize import linprog

# Optional internet weather integration uses Open-Meteo (free, no API key).
try:
    import requests
except ImportError:
    requests = None

st.set_page_config(
    page_title="AgriBloom | AI Crop Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ml" / "crop_model.pkl"

# DEMO PLACEHOLDERS ONLY. These are not verified local agronomic values.
RESOURCE_DATA = {
    "rice": {"water": 4000, "cost": 8000, "benefit": 12000},
    "maize": {"water": 2500, "cost": 6000, "benefit": 9000},
    "chickpea": {"water": 1500, "cost": 4000, "benefit": 7000},
    "kidneybeans": {"water": 1800, "cost": 4500, "benefit": 6500},
    "mothbeans": {"water": 1400, "cost": 3500, "benefit": 5500},
    "mungbean": {"water": 1600, "cost": 3800, "benefit": 6000},
    "pigeonpeas": {"water": 2000, "cost": 4200, "benefit": 6500},
    "blackgram": {"water": 1600, "cost": 3700, "benefit": 5800},
    "lentil": {"water": 1700, "cost": 3900, "benefit": 6000},
    "cotton": {"water": 3000, "cost": 6500, "benefit": 9500},
    "jute": {"water": 2800, "cost": 6000, "benefit": 8500},
    "coffee": {"water": 2500, "cost": 7000, "benefit": 10000},
    "banana": {"water": 3500, "cost": 9000, "benefit": 13000},
    "mango": {"water": 2200, "cost": 6500, "benefit": 9500},
    "grapes": {"water": 2500, "cost": 8500, "benefit": 12000},
    "pomegranate": {"water": 1800, "cost": 7000, "benefit": 10000},
    "orange": {"water": 2300, "cost": 7500, "benefit": 10500},
    "papaya": {"water": 2800, "cost": 8000, "benefit": 11500},
    "coconut": {"water": 3000, "cost": 7500, "benefit": 10000},
    "apple": {"water": 2200, "cost": 9000, "benefit": 12000},
    "watermelon": {"water": 2000, "cost": 5000, "benefit": 8000},
    "muskmelon": {"water": 1800, "cost": 4800, "benefit": 7500},
}

# GENERAL EXAMPLE CALENDAR ONLY. Verify sowing windows with local agriculture experts.
CROP_CALENDAR = {
    "rice": ("Nursery / sowing", "Transplanting / establishment", "Vegetative growth", "Flowering and grain filling", "Harvest"),
    "maize": ("Sowing", "Seedling establishment", "Vegetative growth", "Flowering and grain filling", "Harvest"),
    "chickpea": ("Sowing", "Seedling establishment", "Branching and flowering", "Pod development", "Harvest"),
    "cotton": ("Sowing", "Establishment", "Vegetative growth", "Flowering and boll development", "Picking / harvest"),
    "mungbean": ("Sowing", "Establishment", "Vegetative growth", "Flowering and pod filling", "Harvest"),
    "pigeonpeas": ("Sowing", "Establishment", "Vegetative growth", "Flowering and pod development", "Harvest"),
    "blackgram": ("Sowing", "Establishment", "Vegetative growth", "Flowering and pod filling", "Harvest"),
    "jute": ("Sowing", "Establishment", "Stem growth", "Stem development", "Harvest"),
    "banana": ("Planting", "Establishment", "Leaf and stem growth", "Bunch development", "Harvest"),
    "tomato": ("Nursery / planting", "Establishment", "Vegetative growth", "Flowering and fruit set", "Harvest"),
}
DEFAULT_STAGES = ("Planting / sowing", "Establishment", "Vegetative growth", "Flowering / fruit development", "Harvest")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_crops(model, features):
    frame = pd.DataFrame([features])
    probs = model.predict_proba(frame)[0]
    return sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)


def optimize_plan(ranked, land, water, budget, threshold=0.05):
    selected = [
        (crop, float(prob))
        for crop, prob in ranked
        if prob >= threshold and crop in RESOURCE_DATA
    ][:3]
    if not selected:
        return selected, None, None

    crops = [c for c, _ in selected]
    data = [RESOURCE_DATA[c] for c in crops]
    # Demo objective only: illustrative benefit weighted by model probability.
    objective = [
        -(item["benefit"] * prob)
        for item, (crop, prob) in zip(data, selected)
    ]
    result = linprog(
        c=objective,
        A_ub=[
            [1] * len(crops),
            [item["water"] for item in data],
            [item["cost"] for item in data],
        ],
        b_ub=[land, water, budget],
        bounds=[(0, None)] * len(crops),
        method="highs",
    )
    return selected, data, result


def plan_table(selected, data, result):
    rows = []
    for i, (crop, prob) in enumerate(selected):
        acres = float(result.x[i])
        rows.append({
            "Crop": crop.title(),
            "Model probability (%)": round(prob * 100, 2),
            "Allocated land (acres)": round(acres, 3),
            "Water used (demo units)": round(acres * data[i]["water"], 2),
            "Budget used (₹)": round(acres * data[i]["cost"], 2),
            "Illustrative benefit (₹)": round(acres * data[i]["benefit"], 2),
        })
    return pd.DataFrame(rows)


def get_weather(city):
    """Resolve a place via Open-Meteo geocoding, then fetch daily forecast."""
    if requests is None:
        raise RuntimeError("The requests package is not installed.")
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=12,
    )
    geo.raise_for_status()
    places = geo.json().get("results", [])
    if not places:
        raise ValueError("Place not found. Try a nearby town or city name.")
    place = places[0]
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 7,
        },
        timeout=12,
    )
    weather.raise_for_status()
    return place, weather.json().get("daily", {})


# Nature-inspired visual styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root { --cream:#F6F5ED; --sage:#55745A; --forest:#294A35; --muted:#71806D; --line:#E2E5D8; }
.stApp { background:var(--cream); color:var(--forest); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1250px; padding-top:1.5rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--forest)!important; font-family:'Playfair Display',Georgia,serif!important; }
p,label,.stMarkdown { color:#485B48; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:17px; border-radius:15px; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--forest); font-weight:700; }
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.2rem; }
.stButton>button,.stFormSubmitButton>button { background:var(--sage); color:white; border:1px solid var(--sage); border-radius:10px; min-height:43px; font-weight:600; }
.stButton>button:hover,.stFormSubmitButton>button:hover { background:var(--forest); color:white; border-color:var(--forest); }
.hero { background:linear-gradient(120deg,#E5ECDD,#F1F0E3); border:1px solid #DCE4D4; border-radius:22px; padding:30px; margin:10px 0 22px; }
.hero-tag { color:#55745A; font-size:.76rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.hero-title { color:#294A35; font-family:'Playfair Display',Georgia,serif; font-size:clamp(2rem,4vw,3.2rem); line-height:1.15; font-weight:700; margin:12px 0; }
.hero-copy { color:#5F705A; font-size:1.02rem; line-height:1.7; max-width:750px; }
.kicker { color:#71806D; font-size:.75rem; font-weight:700; letter-spacing:1.7px; text-transform:uppercase; }
.small-note { color:#71806D; font-size:.84rem; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-tag">🌿 Sustainable agriculture · AI insights</div>
  <div class="hero-title">Grow smarter.<br>Plan sustainably.</div>
  <div class="hero-copy">Explore crop recommendations, understand your model,
  compare resource scenarios, check forecast context, and download a farm-planning report.</div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Prototype notice: crop water, cost, and benefit figures are fictional demo assumptions. "
    "The optimizer's benefit is not verified profit or a farm-income forecast. "
    "Use verified regional agronomic data before making real farming decisions."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}: {e}")
    st.stop()

# Persist latest results between Streamlit reruns
if "agri_results" not in st.session_state:
    st.session_state.agri_results = None

st.markdown('<div class="kicker">01 / Farm conditions</div>', unsafe_allow_html=True)
st.markdown("### Tell us about your farm")
with st.form("farm_inputs"):
    soil, weather, resources = st.columns(3)
    with soil:
        st.markdown("#### 🪴 Soil health")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0, step=1.0)
        potassium = st.number_input("Potassium (K)", min_value=0.0, value=42.0, step=1.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    with weather:
        st.markdown("#### ☀️ Weather measurements")
        temperature = st.number_input("Temperature (°C)", value=20.88, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=203.0, step=1.0)
    with resources:
        st.markdown("#### 💧 Resources")
        land = st.number_input("Available land (acres)", min_value=0.01, value=5.0, step=0.5)
        water = st.number_input("Available water (demo units)", min_value=0.0, value=12000.0, step=500.0)
        budget = st.number_input("Available budget (₹)", min_value=0.0, value=30000.0, step=1000.0)
    run_plan = st.form_submit_button("🌾 Generate crop plan", use_container_width=True)

if run_plan:
    features = {
        "N": nitrogen, "P": phosphorus, "K": potassium,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall,
    }
    ranked = predict_crops(model, features)
    selected, data, result = optimize_plan(ranked, land, water, budget)
    allocation = None
    if selected and result is not None and result.success:
        allocation = plan_table(selected, data, result)
    st.session_state.agri_results = {
        "features": features, "ranked": ranked, "selected": selected,
        "data": data, "result": result, "allocation": allocation,
        "land": land, "water": water, "budget": budget,
    }

saved = st.session_state.agri_results
if saved:
    features = saved["features"]
    ranked = saved["ranked"]
    allocation = saved["allocation"]
    st.divider()
    st.markdown('<div class="kicker">02 / AI insights</div>', unsafe_allow_html=True)
    st.markdown("### Crop predictions")
    top3 = ranked[:3]
    c1, c2, c3 = st.columns(3)
    c1.metric("Top predicted crop", top3[0][0].title())
    c2.metric("Model probability", f"{top3[0][1] * 100:.2f}%")
    c3.metric("Predictions shown", len(top3))
    pred_df = pd.DataFrame([{"Crop": c.title(), "Model probability (%)": round(p * 100, 2)} for c, p in top3])
    left, right = st.columns([1, 1.5])
    with left:
        st.dataframe(pred_df, hide_index=True, use_container_width=True)
    with right:
        fig = px.bar(pred_df, x="Model probability (%)", y="Crop", orientation="h",
                     text="Model probability (%)", color_discrete_sequence=["#68866A"],
                     title="Top crop predictions")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#405541", margin=dict(l=10, r=10, t=50, b=10),
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # FEATURE 1: EXPLAINABLE AI
    st.divider()
    st.markdown('<div class="kicker">03 / Explainable AI</div>', unsafe_allow_html=True)
    st.markdown("### What influences the model?")
    st.write(
        "This chart shows global feature importance from the Random Forest model. "
        "It describes how the model used features across training trees; it does not prove causation "
        "or explain a single prediction by itself."
    )
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Input feature": list(features.keys()),
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            importance_df, x="Importance", y="Input feature", orientation="h",
            title="Random Forest feature importance",
            color_discrete_sequence=["#55745A"],
        )
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#405541", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("This model does not expose feature_importances_.")

    st.markdown("#### Input values used for this prediction")
    st.dataframe(pd.DataFrame([features]), hide_index=True, use_container_width=True)

    # FEATURE 2: WHAT-IF SIMULATOR
    st.divider()
    st.markdown('<div class="kicker">04 / What-if simulator</div>', unsafe_allow_html=True)
    st.markdown("### Explore different resource scenarios")
    st.caption("These scenarios reuse the same fictional resource table. They do not predict actual farm profits.")
    with st.form("what_if_form"):
        w1, w2, w3 = st.columns(3)
        with w1:
            alt_land = st.number_input("Scenario land (acres)", min_value=0.01, value=float(saved["land"]), step=0.5, key="alt_land")
        with w2:
            alt_water = st.number_input("Scenario water (demo units)", min_value=0.0, value=float(saved["water"]), step=500.0, key="alt_water")
        with w3:
            alt_budget = st.number_input("Scenario budget (₹)", min_value=0.0, value=float(saved["budget"]), step=1000.0, key="alt_budget")
        run_what_if = st.form_submit_button("Compare scenario", use_container_width=True)

    if run_what_if:
        alt_selected, alt_data, alt_result = optimize_plan(ranked, alt_land, alt_water, alt_budget)
        if not alt_selected:
            st.warning("No crops passed the model-probability threshold.")
        elif not alt_result.success:
            st.error(f"Scenario optimizer failed: {alt_result.message}")
        else:
            alt_df = plan_table(alt_selected, alt_data, alt_result)
            st.markdown("#### Scenario allocation")
            st.dataframe(alt_df, hide_index=True, use_container_width=True)
            current_land = float(allocation["Allocated land (acres)"].sum()) if allocation is not None else 0.0
            current_cost = float(allocation["Budget used (₹)"].sum()) if allocation is not None else 0.0
            current_water = float(allocation["Water used (demo units)"].sum()) if allocation is not None else 0.0
            alt_land_used = float(alt_df["Allocated land (acres)"].sum())
            alt_cost_used = float(alt_df["Budget used (₹)"].sum())
            alt_water_used = float(alt_df["Water used (demo units)"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Scenario land used", f"{alt_land_used:.2f} ac", delta=f"{alt_land_used-current_land:+.2f} ac vs current")
            m2.metric("Scenario water used", f"{alt_water_used:,.0f}", delta=f"{alt_water_used-current_water:+,.0f} vs current")
            m3.metric("Scenario budget used", f"₹{alt_cost_used:,.0f}", delta=f"₹{alt_cost_used-current_cost:+,.0f} vs current")
            st.caption("Comparison deltas compare allocation totals. Values remain illustrative.")

    # Main optimization results
    st.divider()
    st.markdown('<div class="kicker">05 / Resource plan</div>', unsafe_allow_html=True)
    st.markdown("### Suggested crop allocation")
    if not saved["selected"]:
        st.warning("No crops passed the 5% prototype threshold and resource-table check.")
    elif saved["result"] is None or not saved["result"].success:
        st.error("The optimizer could not calculate an allocation.")
    elif allocation is not None:
        total_land = float(allocation["Allocated land (acres)"].sum())
        total_water = float(allocation["Water used (demo units)"].sum())
        total_cost = float(allocation["Budget used (₹)"].sum())
        total_benefit = float(allocation["Illustrative benefit (₹)"].sum())
        a, b, c, d = st.columns(4)
        a.metric("Land allocated", f"{total_land:.2f} acres")
        b.metric("Water used", f"{total_water:,.0f}")
        c.metric("Budget used", f"₹{total_cost:,.0f}")
        d.metric("Illustrative benefit", f"₹{total_benefit:,.0f}")
        st.dataframe(allocation, hide_index=True, use_container_width=True)
        lcol, rcol = st.columns(2)
        with lcol:
            nonzero = allocation[allocation["Allocated land (acres)"] > 0]
            if not nonzero.empty:
                pie = px.pie(nonzero, names="Crop", values="Allocated land (acres)",
                             hole=0.5, title="Land allocation",
                             color_discrete_sequence=["#55745A", "#9CAF88", "#D2DDBE"])
                pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#405541")
                st.plotly_chart(pie, use_container_width=True)
        with rcol:
            usage = pd.DataFrame({
                "Resource": ["Land", "Water", "Budget"],
                "Used": [total_land, total_water, total_cost],
                "Unused": [max(0, saved["land"]-total_land),
                           max(0, saved["water"]-total_water),
                           max(0, saved["budget"]-total_cost)],
            })
            fig_use = px.bar(usage, x="Resource", y=["Used", "Unused"], barmode="stack",
                             title="Used vs unused resources",
                             color_discrete_sequence=["#55745A", "#DDE5D2"])
            fig_use.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#405541")
            st.plotly_chart(fig_use, use_container_width=True)

    # FEATURE 3: REPORT EXPORT (HTML and CSV, no PDF dependency)
    st.divider()
    st.markdown('<div class="kicker">06 / Farm report</div>', unsafe_allow_html=True)
    st.markdown("### Download your results")
    if allocation is not None:
        report_dict = {
            "report_title": "AgriBloom Crop Planning Report",
            "generated_on": date.today().isoformat(),
            "input_measurements": features,
            "top_predictions": [
                {"crop": crop, "model_probability_percent": round(float(prob)*100, 2)}
                for crop, prob in ranked[:3]
            ],
            "resource_constraints": {
                "land_acres": saved["land"],
                "water_demo_units": saved["water"],
                "budget_rupees": saved["budget"],
            },
            "allocation": allocation.to_dict(orient="records"),
            "disclaimer": "Water, cost, and benefit are fictional demonstration assumptions, not verified farm advice or profit estimates.",
        }
        report_json = json.dumps(report_dict, indent=2, ensure_ascii=False)
        st.download_button(
            "Download complete report (JSON)",
            data=report_json.encode("utf-8"),
            file_name="agribloom_farm_report.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Download allocation table (CSV)",
            data=allocation.to_csv(index=False).encode("utf-8"),
            file_name="agribloom_allocation.csv",
            mime="text/csv",
            use_container_width=True,
        )
        report_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>AgriBloom Farm Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1000px;margin:35px auto;padding:0 20px;color:#294A35;background:#F6F5ED}}
h1,h2{{color:#294A35}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #dce4d4;padding:9px;text-align:left}}
.notice{{background:#fff0c2;padding:15px;border-radius:8px}}</style></head><body>
<h1>AgriBloom Crop Planning Report</h1><p>Generated: {date.today().isoformat()}</p>
<h2>Input measurements</h2><table><tr><th>Feature</th><th>Value</th></tr>
{''.join(f'<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>' for k,v in features.items())}</table>
<h2>Top crop predictions</h2><table><tr><th>Crop</th><th>Model probability</th></tr>
{''.join(f'<tr><td>{html.escape(str(crop.title()))}</td><td>{prob*100:.2f}%</td></tr>' for crop,prob in ranked[:3])}</table>
<h2>Allocation</h2>{allocation.to_html(index=False, escape=True)}
<div class="notice"><b>Important:</b> Water, cost and benefit values are fictional demonstration assumptions.
The benefit is not verified profit or a farm-income forecast. Model probability is not a guarantee of crop success.</div>
</body></html>"""
        st.download_button(
            "Download printable report (HTML)",
            data=report_html.encode("utf-8"),
            file_name="agribloom_farm_report.html",
            mime="text/html",
            use_container_width=True,
        )
    else:
        st.info("Generate a crop plan with a feasible allocation to enable report downloads.")

    # FEATURE 4: WEATHER-AWARE PLANNING
    st.divider()
    st.markdown('<div class="kicker">07 / Weather context</div>', unsafe_allow_html=True)
    st.markdown("### Check a local 7-day forecast")
    st.caption("Weather is contextual information only; it is not currently fed into the trained crop model or optimizer.")
    with st.form("weather_form"):
        city = st.text_input("Town or city", value="Hyderabad")
        weather_submit = st.form_submit_button("Fetch free forecast", use_container_width=True)
    if weather_submit:
        try:
            place, daily = get_weather(city.strip())
            if not daily or not daily.get("time"):
                st.warning("No daily forecast was returned for this place.")
            else:
                st.success(
                    f"Forecast location: {place.get('name')}, "
                    f"{place.get('admin1', '')}, {place.get('country', '')}"
                )
                wx_df = pd.DataFrame({
                    "Date": daily["time"],
                    "Maximum temperature (°C)": daily.get("temperature_2m_max", []),
                    "Minimum temperature (°C)": daily.get("temperature_2m_min", []),
                    "Rainfall (mm)": daily.get("precipitation_sum", []),
                    "Rain probability (%)": daily.get("precipitation_probability_max", []),
                })
                st.dataframe(wx_df, hide_index=True, use_container_width=True)
                wx_long = wx_df.melt(
                    id_vars="Date",
                    value_vars=["Maximum temperature (°C)", "Minimum temperature (°C)"],
                    var_name="Measure", value_name="Temperature (°C)"
                )
                fig_weather = px.line(wx_long, x="Date", y="Temperature (°C)", color="Measure",
                                      markers=True, title="Forecast temperatures",
                                      color_discrete_sequence=["#55745A", "#D39A58"])
                fig_weather.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#405541")
                st.plotly_chart(fig_weather, use_container_width=True)
                st.caption("Source: Open-Meteo forecast and geocoding services. Forecasts can change; check local official advisories.")
        except Exception as e:
            st.error(f"Could not fetch weather. Check your internet connection and place name. Details: {e}")

    # FEATURE 5: CROP CALENDAR
    st.divider()
    st.markdown('<div class="kicker">08 / Crop calendar</div>', unsafe_allow_html=True)
    st.markdown("### General crop-stage planner")
    st.caption("Illustrative stages only, not a region-specific sowing calendar. Timing varies by variety, climate, irrigation, and local recommendations.")
    options = sorted(set([c for c, _ in ranked[:3]] + list(CROP_CALENDAR.keys())))
    selected_calendar_crop = st.selectbox(
        "Choose a crop", options, format_func=lambda x: x.title(), key="calendar_crop"
    )
    stages = CROP_CALENDAR.get(selected_calendar_crop, DEFAULT_STAGES)
    start_date = st.date_input("Approximate planting date", value=date.today(), key="planting_date")
    # Generic visual timeline spacing; not a biological duration estimate.
    timeline_rows = []
    for i, stage in enumerate(stages):
        stage_date = start_date + timedelta(days=i * 20)
        timeline_rows.append({
            "Stage": stage,
            "Illustrative date": stage_date.isoformat(),
            "Note": "Placeholder interval — verify locally",
        })
    st.dataframe(pd.DataFrame(timeline_rows), hide_index=True, use_container_width=True)
    st.info(
        "The 20-day spacing is a visual placeholder, not a claim about actual crop duration. "
        "Replace it with locally verified crop-calendar dates before using it for farm decisions."
    )

else:
    st.info("Enter farm measurements and select **Generate crop plan** to unlock predictions, explainability, scenario comparison, and reports.")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#71806D;font-size:.82rem">'
    '🌿 AgriBloom · AI-assisted crop planning prototype · Verify local agronomic data</div>',
    unsafe_allow_html=True,
)
