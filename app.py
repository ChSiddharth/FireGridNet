import os
import json
import pandas as pd
import tempfile
import database
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from PIL import Image, ExifTags
from ultralytics import YOLO

st.set_page_config(
    page_title="FireGridNet | Wildfire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)
database.initialize_database()
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

st.markdown("""
<style>
.stApp {
    background: #101114;
    color: #F2F2F2;
}
[data-testid="stSidebar"] {
    background: #191A1F;
    border-right: 1px solid #303137;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
h1, h2, h3 {
    color: #F5F5F5 !important;
}
p, label, .stMarkdown {
    color: #D2D2D7;
}
[data-testid="stMetric"] {
    background: #1C1D23;
    border: 1px solid #35363D;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #B6B7C0;
}
[data-testid="stMetricValue"] {
    color: #FF8A45;
}
.stButton > button {
    background: #F47735;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
}
.stButton > button:hover {
    background: #FF965C;
    color: white;
    border: none;
}
hr {
    border-color: #35363D;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
.small-muted {
    color: #A6A7B0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


def get_image_gps(image):
    try:
        exif = image.getexif()
        gps_tag = next(
            (tag for tag, name in ExifTags.TAGS.items()
             if name == "GPSInfo"),
            None
        )

        if not gps_tag or gps_tag not in exif:
            return None

        gps = exif.get_ifd(gps_tag)

        def decimal(value):
            return float(value[0]) / float(value[1])

        def coordinate(value):
            return (
                decimal(value[0])
                + decimal(value[1]) / 60
                + decimal(value[2]) / 3600
            )

        lat = coordinate(gps[2])
        lon = coordinate(gps[4])

        if gps[1] == "S":
            lat = -lat
        if gps[3] == "W":
            lon = -lon

        return lat, lon
    except Exception:
        return None


def find_location(place):
    response = requests.get(
        GEOCODE_URL,
        params={"name": place, "count": 5, "language": "en"},
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def get_weather(lat, lon):
    response = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation,wind_speed_10m"
            ),
            "hourly": "precipitation",
            "past_hours": 24,
            "forecast_hours": 1,
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    current = data["current"]
    hourly = data.get("hourly", {})
    rain_values = [
        value for value in hourly.get("precipitation", [])
        if value is not None
    ]

    return {
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "wind": current["wind_speed_10m"],
        "rain_now": current["precipitation"],
        "rain_24h": sum(rain_values),
        "time": current.get("time", "Unknown"),
        "timezone": data.get("timezone", "Local time"),
    }


def calculate_risk(temperature, humidity, rainfall, wind):
    score = (
        0.35 * max(0, min(100, temperature / 60 * 100))
        + 0.30 * (100 - humidity)
        + 0.20 * (min(wind, 100) / 100 * 100)
        + 0.15 * (
            100 - min(rainfall, 50) / 50 * 100
        )
    )
    score = max(0, min(100, score))

    if score < 30:
        category = "Lower"
    elif score < 60:
        category = "Moderate"
    else:
        category = "Higher"

    return score, category


def risk_color(score):
    if score < 30:
        return "#2EAD66"
    if score < 60:
        return "#F2B705"
    return "#E34B4B"


def show_risk_breakdown(temperature, humidity, rainfall, wind, score, category):
    temp_part = 0.35 * max(0, min(100, temperature / 60 * 100))
    humidity_part = 0.30 * (100 - humidity)
    wind_part = 0.20 * (min(wind, 100) / 100 * 100)
    rain_part = 0.15 * (100 - min(rainfall, 50) / 50 * 100)
    st.subheader("Risk Score Breakdown")
    st.caption(
        "Contributions from the prototype formula, not a trained or "
        "operationally validated wildfire prediction model."
    )
    breakdown = pd.DataFrame({
        "Weather factor": ["Temperature", "Low humidity", "Wind speed", "Low rainfall"],
        "Score contribution": [temp_part, humidity_part, wind_part, rain_part],
    }).set_index("Weather factor")
    st.bar_chart(breakdown)
    st.progress(int(score))
    st.metric("Experimental risk score", f"{score:.1f}/100", category)
    if category == "Higher":
        st.error("HIGHER-RISK SCENARIO: verify conditions with official local sources.")
    elif category == "Moderate":
        st.warning("MODERATE-RISK SCENARIO: continue monitoring weather and local reports.")
    else:
        st.success("LOWER-RISK SCENARIO: this estimate does not prove an area is safe.")


def get_record_type(incident):
    name = incident.get("image_name") or ""
    if name == "Weather Risk Assessment":
        return "Weather Risk Assessment"
    if name == "Manual Risk Assessment":
        return "Manual Risk Assessment"
    return "AI Detection"


# Sidebar navigation
with st.sidebar:
    st.markdown("# 🔥 FireGridNet")
    st.caption("WILDFIRE INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "AI Detection",
            "Risk Assessment",
            "Incident Map",
            "Incident History",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown("🟢 Application interface loaded")
    st.markdown("🟠 AI model loads when required")
    st.caption("Academic research prototype")


# Header
st.markdown(
    '<p class="small-muted">WILDFIRE MONITORING / AI-POWERED ANALYSIS</p>',
    unsafe_allow_html=True,
)
st.title("🔥 FireGridNet")
st.caption(
    "AI-Powered Wildfire Detection & Risk Assessment"
)
st.divider()


if page == "Overview":
    st.subheader("Monitoring Overview")
    st.write(
        "A unified workspace for visual fire detection, "
        "weather analysis and wildfire incident monitoring."
    )

    a, b, c = st.columns(3)
    a.metric("AI Model", "YOLO", "Fire + Smoke classes")
    b.metric("Weather Source", "Open-Meteo", "Location-based estimates")
    c.metric("Risk Engine", "Prototype", "Not operationally validated")

    st.divider()
    st.subheader("Monitoring Modules")

    left, right = st.columns(2)
    with left:
        st.markdown("### 🔥 Visual Detection")
        st.write(
            "Analyse an uploaded image for possible fire and smoke, "
            "with bounding boxes and confidence scores."
        )
        if st.button("Open AI Detection", use_container_width=True):
            st.session_state["navigation_target"] = "AI Detection"
            st.rerun()

        st.markdown("### 🌦️ Environmental Risk")
        st.write(
            "Retrieve weather for a selected location and view "
            "an experimental weather-based risk estimate."
        )
        if st.button("Open Risk Assessment", use_container_width=True):
            st.session_state["navigation_target"] = "Risk Assessment"
            st.rerun()

    with right:
        st.markdown("### 🗺️ Incident Geography")
        st.write(
            "Explore the H3 forest grid, saved observation markers, "
            "and a clearly labelled prototype risk overlay."
        )
        if st.button("Open Incident Map", use_container_width=True):
            st.session_state["navigation_target"] = "Incident Map"
            st.rerun()

        st.markdown("### 📋 Incident Records")
        st.write(
            "Review saved AI detections and weather assessments, "
            "filter records, and update their review status."
        )
        if st.button("Open Incident History", use_container_width=True):
            st.session_state["navigation_target"] = "Incident History"
            st.rerun()

    st.divider()
    st.subheader("🚨 Monitoring Alerts")
    saved_incidents = database.get_incidents()
    high_risk_records = [
        item for item in saved_incidents
        if item.get("risk_score") is not None and float(item["risk_score"]) >= 60
    ]
    unreviewed_detections = []
    for item in saved_incidents:
        try:
            objects = json.loads(item.get("detections_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            objects = []
        if objects and item.get("status", "New") == "New":
            unreviewed_detections.append(item)
    alert_a, alert_b = st.columns(2)
    alert_a.metric("Higher-risk assessments", len(high_risk_records))
    alert_b.metric("New AI detection records", len(unreviewed_detections))
    if high_risk_records:
        st.warning("Review higher-risk weather assessments in Incident History. These are experimental estimates.")
    if unreviewed_detections:
        st.info("New AI detection records need review; detections are not confirmed wildfire reports.")
    if not high_risk_records and not unreviewed_detections:
        st.success("No saved prototype alerts currently meet these rules.")
    st.caption("Alerts use saved app records only; no emergency service or live wildfire feed is connected.")


elif page == "AI Detection":
    st.subheader("AI Fire & Smoke Detection")
    st.write(
        "Upload an image to inspect possible visible fire or smoke."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="detection_upload",
    )


    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_gps = get_image_gps(image)
        image = image.convert("RGB")

        st.markdown("### 📍 Observation Location")

        location_choice = st.radio(
            "How would you like to set the location?",
            ["Enter coordinates", "Use image GPS"],
            horizontal=True,
            key="detection_location_choice",
        )

        observation_lat = None
        observation_lon = None
        location_source = None

        if location_choice == "Enter coordinates":
            col_lat, col_lon = st.columns(2)

            with col_lat:
                observation_lat = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=17.3850,
                    format="%.5f",
                    key="detection_latitude",
                )

            with col_lon:
                observation_lon = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=78.4867,
                    format="%.5f",
                    key="detection_longitude",
                )

            location_source = "Manually entered coordinates"

        elif image_gps:
            observation_lat, observation_lon = image_gps
            location_source = "Image EXIF GPS"

            st.success(
                f"Using image GPS: {observation_lat:.5f}, "
                f"{observation_lon:.5f}"
            )
        else:
            st.warning(
                "This image has no readable GPS metadata. "
                "Choose Enter coordinates to assign a location."
            )

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )



    if st.button("Run AI Analysis", type="primary"):
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp:
                image.save(temp.name)
                temp_path = temp.name

            with st.spinner("Analysing image with YOLO..."):
                model = load_model()
                result = model.predict(
                    source=temp_path,
                    conf=0.25,
                    verbose=False,
                )[0]

            plotted = result.plot()[:, :, ::-1]
            st.subheader("Detection Results")
            st.image(
                plotted,
                caption="AI predictions",
                use_container_width=True,
            )

            detections = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 4),
                })

                st.write(
                    f"**{class_name.title()}** — "
                    f"{confidence:.1%} confidence"
                )

            if detections:
                database.save_incident(
                    image_name=uploaded_image.name,
                    detections=detections,
                    latitude=observation_lat,
                    longitude=observation_lon,
                    location_source=location_source,
                )

                st.success(
                    "Detection results saved to incident history."
                )
            else:
                st.warning(
                    "No fire or smoke detected above the current "
                    "confidence threshold. This does not prove "
                    "that the area is safe. No incident was saved."
                )

        except Exception as error:
            st.error(f"Image analysis failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)



elif page == "Risk Assessment":
    st.subheader("Environmental Risk Assessment")
    st.caption(
        "Weather data is model-based and may not represent conditions "
        "at the exact forest site. The current score is illustrative."
    )

    mode = st.radio(
        "Weather input",
        ["Fetch automatically", "Enter manually"],
        horizontal=True,
    )

    lat = None
    lon = None
    selected_location = None

    if mode == "Fetch automatically":
        location_mode = st.radio(
            "Location method",
            ["Search for a place", "Enter coordinates", "Image GPS"],
            horizontal=True,
        )

        if location_mode == "Search for a place":
            place = st.text_input(
                "Search for a forest, park, town or place"
            )

            if st.button("Find Location"):
                if place.strip():
                    try:
                        places = find_location(place.strip())
                        st.session_state["places"] = places
                        if not places:
                            st.warning("No matching places found.")
                    except requests.RequestException as error:
                        st.error(f"Location search failed: {error}")
                else:
                    st.warning("Enter a place name first.")

            places = st.session_state.get("places", [])
            if places:
                choices = [
                    f"{p['name']}, {p.get('admin1', '')}, "
                    f"{p.get('country', '')} "
                    f"({p['latitude']:.3f}, {p['longitude']:.3f})"
                    for p in places
                ]
                chosen = st.selectbox("Select matching place", choices)
                selected_place = places[choices.index(chosen)]
                lat = selected_place["latitude"]
                lon = selected_place["longitude"]
                selected_location = chosen

        elif location_mode == "Enter coordinates":
            lat = st.number_input(
                "Latitude", -90.0, 90.0, 20.0, format="%.5f"
            )
            lon = st.number_input(
                "Longitude", -180.0, 180.0, 78.0, format="%.5f"
            )
            selected_location = "Manually entered coordinates"

        else:
            st.info(
                "Image GPS is available when the uploaded image contains "
                "readable GPS metadata. Use AI Detection to inspect an image."
            )

        if st.button("Fetch Current Weather", type="primary"):
            if lat is None or lon is None:
                st.warning("Select a location first.")
            else:
                try:
                    weather = get_weather(lat, lon)
                    st.session_state["weather"] = weather
                    st.session_state["weather_coords"] = (lat, lon)
                    st.session_state["weather_location"] = selected_location
                except (requests.RequestException, KeyError, ValueError) as error:
                    st.error(f"Could not retrieve weather: {error}")

        weather = st.session_state.get("weather")
        saved_coords = st.session_state.get("weather_coords")
        current_coords = (
            (lat, lon) if lat is not None and lon is not None else None
        )

        if weather and saved_coords == current_coords and current_coords:
            st.success(
                f"Weather for {st.session_state.get('weather_location', 'selected location')}"
            )
            st.caption(
                f"Weather time: {weather['time']} "
                f"({weather['timezone']})"
            )

            a, b = st.columns(2)
            a.metric("Temperature", f"{weather['temperature']} °C")
            b.metric("Humidity", f"{weather['humidity']}%")
            a.metric("Wind Speed", f"{weather['wind']} km/h")
            b.metric(
                "Current Precipitation",
                f"{weather['rain_now']} mm",
            )
            st.metric(
                "Precipitation in returned window",
                f"{weather['rain_24h']:.1f} mm",
            )

            
            if st.button("Calculate Experimental Risk"):
                score, category = calculate_risk(
                    weather["temperature"],
                    weather["humidity"],
                    weather["rain_24h"],
                    weather["wind"],
                )

                st.session_state["latest_risk_score"] = score
                st.session_state["latest_risk_category"] = category
                st.session_state["latest_risk_weather"] = {
                    "temperature": weather["temperature"],
                    "humidity": weather["humidity"],
                    "rainfall": weather["rain_24h"],
                    "wind": weather["wind"],
                }
                show_risk_breakdown(
                    weather["temperature"], weather["humidity"],
                    weather["rain_24h"], weather["wind"], score, category
                )

                database.save_incident(
                    image_name="Weather Risk Assessment",
                    detections=[],
                    latitude=lat,
                    longitude=lon,
                    location_source=selected_location,
                    temperature=weather["temperature"],
                    humidity=weather["humidity"],
                    wind_speed=weather["wind"],
                    rainfall_24h=weather["rain_24h"],
                    risk_score=score,
                    risk_category=category,
                )

                st.success("Risk assessment saved to incident history.")


    else:
        temperature = st.number_input(
            "Temperature (°C)", -10.0, 60.0, 35.0
        )
        humidity = st.slider("Relative humidity (%)", 0, 100, 30)
        rainfall = st.number_input(
            "Rainfall in last 24 hours (mm)", 0.0, 500.0, 0.0
        )
        wind = st.number_input(
            "Wind speed (km/h)", 0.0, 200.0, 20.0
        )

        
        if st.button("Calculate Experimental Risk", type="primary"):
            score, category = calculate_risk(
                temperature, humidity, rainfall, wind
            )

            st.session_state["latest_risk_score"] = score
            st.session_state["latest_risk_category"] = category
            st.session_state["latest_risk_weather"] = {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "wind": wind,
            }
            show_risk_breakdown(
                temperature, humidity, rainfall, wind, score, category
            )

            database.save_incident(
                image_name="Manual Risk Assessment",
                detections=[],
                location_source="Manual weather inputs",
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind,
                rainfall_24h=rainfall,
                risk_score=score,
                risk_category=category,
            )

            st.success("Manual risk assessment saved to incident history.")




    st.subheader("Wildfire Observation Map")
elif page == "Incident Map":
    st.subheader("🗺️ GIS Incident & Grid Risk Map")
    st.write(
        "Explore the H3 forest grid and saved observations. The risk overlay "
        "uses the latest experimental weather score as a scenario for every cell; "
        "it is not a separate prediction for each cell."
    )

    incidents = database.get_incidents()
    mapped_incidents = [
        incident for incident in incidents
        if incident.get("latitude") is not None
        and incident.get("longitude") is not None
    ]

    latest_score = st.session_state.get("latest_risk_score")
    latest_category = st.session_state.get("latest_risk_category", "Not calculated")
    st.subheader("Grid-Based Risk Scenario")
    if latest_score is None:
        st.info(
            "Calculate a weather risk on the Risk Assessment page first. "
            "Until then, the grid shows forest cells without a risk classification."
        )
        grid_score = None
    else:
        st.metric("Latest scenario score", f"{latest_score:.1f}/100", latest_category)
        st.caption(
            "Prototype limitation: this score is applied uniformly to the displayed "
            "forest grid. Per-cell weather retrieval and a trained local model are future work."
        )
        grid_score = float(latest_score)

    geojson_path = "forest_grid_v4.geojson"
    if os.path.exists(geojson_path):
        try:
            with open(geojson_path, "r", encoding="utf-8") as file:
                forest_data = json.load(file)

            forest_map = folium.Map(
                location=[35.0, 85.0], zoom_start=5,
                tiles="OpenStreetMap", control_scale=True,
            )

            def grid_style(feature):
                fill = "#228B22" if grid_score is None else risk_color(grid_score)
                return {
                    "fillColor": fill, "color": "#E7E7E7",
                    "weight": 1, "fillOpacity": 0.42 if grid_score is None else 0.55,
                }

            first_feature = (forest_data.get("features") or [{}])[0]
            tooltip_fields = [
                field for field in ("cell_id", "forest_sample_points")
                if field in first_feature.get("properties", {})
            ]
            geojson_kwargs = {
                "data": forest_data,
                "name": "Forest grid / risk scenario",
                "style_function": grid_style,
            }
            if tooltip_fields:
                geojson_kwargs["tooltip"] = folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=[
                        "H3 Cell:" if field == "cell_id" else "Forest Sample Points:"
                        for field in tooltip_fields
                    ],
                    localize=True,
                )
            folium.GeoJson(**geojson_kwargs).add_to(forest_map)

            for incident in mapped_incidents:
                try:
                    detections = json.loads(incident.get("detections_json") or "[]")
                except (TypeError, json.JSONDecodeError):
                    detections = []
                record_type = get_record_type(incident)
                score = incident.get("risk_score")
                if score is not None:
                    marker_color = "red" if float(score) >= 60 else ("orange" if float(score) >= 30 else "green")
                elif detections:
                    marker_color = "red"
                else:
                    marker_color = "blue"
                popup_text = (
                    f"Record #{incident.get('id')}<br>Type: {record_type}<br>"
                    f"Record: {incident.get('image_name') or 'Observation'}<br>"
                    f"Detected objects: {incident.get('detection_count', 0)}<br>"
                    f"Risk score: {score if score is not None else 'Not applicable'}"
                )
                folium.Marker(
                    location=[float(incident["latitude"]), float(incident["longitude"])],
                    tooltip=f"#{incident.get('id')} · {record_type}",
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=marker_color, icon="info-sign"),
                ).add_to(forest_map)

            folium.LayerControl(collapsed=False).add_to(forest_map)
            st_folium(
                forest_map, use_container_width=True, height=560,
                key="firegridnet_gis_map",
            )
            if grid_score is None:
                st.caption("Green cells = forest grid. No risk scenario has been applied yet.")
            else:
                st.caption(
                    "Red = higher prototype score (60–100); amber = moderate (30–59.9); "
                    "green = lower (0–29.9). These colors show a scenario, not verified local risk."
                )
            st.caption(
                "Forest sample counts are not exact forest-coverage percentages. "
                "Only saved observations with coordinates are plotted."
            )
        except Exception as error:
            st.error(f"Could not render the forest grid: {error}")
    else:
        st.warning("Forest grid file not found: forest_grid_v4.geojson")

    st.divider()
    st.subheader("🚨 Alerts & Priority Review")
    higher_risk = [
        item for item in incidents
        if item.get("risk_score") is not None and float(item["risk_score"]) >= 60
    ]
    new_detection_records = []
    for item in incidents:
        try:
            detections = json.loads(item.get("detections_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            detections = []
        if detections and item.get("status", "New") == "New":
            new_detection_records.append((item, detections))

    alert_a, alert_b, alert_c = st.columns(3)
    alert_a.metric("Saved records", len(incidents))
    alert_b.metric("Higher-risk assessments", len(higher_risk))
    alert_c.metric("New detections to review", len(new_detection_records))
    if higher_risk:
        st.markdown("**Higher-risk weather assessments**")
        for item in sorted(higher_risk, key=lambda record: float(record["risk_score"]), reverse=True):
            st.warning(
                f"Record #{item.get('id')} · score {float(item['risk_score']):.1f}/100 · "
                f"{item.get('timestamp', 'time unavailable')}"
            )
    if new_detection_records:
        st.markdown("**AI detections awaiting review**")
        for item, detections in new_detection_records:
            names = ", ".join(
                f"{d.get('class', 'object').title()} ({float(d.get('confidence', 0)):.0%})"
                for d in detections
            )
            st.info(f"Record #{item.get('id')} · {item.get('image_name') or 'Uploaded image'} · {names}")
    if not higher_risk and not new_detection_records:
        st.success("No saved records currently match the prototype alert rules.")
    st.caption(
        "This is an academic prototype alert panel, not a live wildfire alert service. "
        "Verify urgent conditions with official authorities."
    )

    st.subheader("Mapped Observations")
    if mapped_incidents:
        map_rows = [{
            "ID": incident.get("id"),
            "Record Type": get_record_type(incident),
            "Image / Record": incident.get("image_name") or "Uploaded image",
            "Objects Detected": incident.get("detection_count", 0),
            "Risk Score": incident.get("risk_score"),
            "Latitude": float(incident["latitude"]),
            "Longitude": float(incident["longitude"]),
            "Status": incident.get("status", "New"),
        } for incident in mapped_incidents]
        st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No saved observations have coordinates yet. Run an AI analysis or a weather "
            "assessment with coordinates to add a marker."
        )


elif page == "Incident History":
    st.subheader("Incident History")
    st.write(
        "Review fire and smoke detections saved by FireGridNet."
    )

    incidents = database.get_incidents()

    if not incidents:
        st.info(
            "No incidents saved yet. Go to AI Detection, "
            "upload an image, and run AI Analysis."
        )
    else:
        st.metric("Saved Incidents", len(incidents))

        rows = []

        for incident in incidents:
            detections = json.loads(
                incident["detections_json"] or "[]"
            )

            detection_summary = ", ".join(
                f'{item["class"].title()} '
                f'({item["confidence"]:.1%})'
                for item in detections
            )

            rows.append({
                "Incident ID": incident["id"],
                "Timestamp (UTC)": incident["timestamp"],
                "Image": incident["image_name"] or "Unknown",
                "Record Type": (
                    "Weather Risk Assessment"
                     if incident["image_name"] == "Weather Risk Assessment"
                     else "Manual Risk Assessment"
                     if incident["image_name"] == "Manual Risk Assessment"
                     else "AI Detection"
            ),
                "Objects Detected": incident["detection_count"],
                "Detection Details": detection_summary or "None",
                "Latitude": incident["latitude"],
                "Longitude": incident["longitude"],
                "Status": incident["status"],
                "Risk Score": incident["risk_score"],
                "Risk Category": incident["risk_category"],
            })

        
        df = pd.DataFrame(rows)

        st.subheader("Filter Incidents")

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            record_filter = st.selectbox(
                "Record Type",
                [
                    "All Types",
                    "AI Detection",
                    "Weather Risk Assessment",
                    "Manual Risk Assessment",
                ],
            )

        with filter_col2:
            status_filter = st.selectbox(
                "Status",
                [
                    "All Statuses",
                    "New",
                    "Under Review",
                    "Resolved",
                ],
            )

        filtered_df = df.copy()

        if record_filter != "All Types":
            filtered_df = filtered_df[
                filtered_df["Record Type"] == record_filter
            ]

        if status_filter != "All Statuses":
            filtered_df = filtered_df[
                filtered_df["Status"] == status_filter
            ]

        st.caption(
            f"Showing {len(filtered_df)} of {len(df)} incidents"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "AI detections and weather risk estimates are "
            "different record types. Neither alone confirms "
            "an actual wildfire."
        )

        st.subheader("Update Incident Status")

        incident_options = {
            f"Incident #{incident['id']} — "
            f"{incident['image_name'] or 'Uploaded image'}": incident["id"]
            for incident in incidents
        }

        selected_incident_label = st.selectbox(
            "Select an incident",
            options=list(incident_options.keys()),
        )

        selected_incident_id = incident_options[
            selected_incident_label
        ]

        selected_incident = next(
            incident
            for incident in incidents
            if incident["id"] == selected_incident_id
        )

        status_options = ["New", "Under Review", "Resolved"]
        current_status = selected_incident["status"]

        if current_status not in status_options:
            current_status = "New"

        new_status = st.selectbox(
            "Choose status",
            options=status_options,
            index=status_options.index(current_status),
        )

        if st.button("Save Status", type="primary"):
            database.update_incident_status(
                selected_incident_id,
                new_status,
            )
            st.success("Incident status updated.")
            st.rerun()

    if st.button("Refresh Incident History"):
        st.rerun()
