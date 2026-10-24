
import os
import tempfile

import requests
import streamlit as st
from PIL import Image, ExifTags
from ultralytics import YOLO

st.set_page_config(
    page_title="FireGridNet",
    page_icon="🔥",
    layout="wide"
)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


def get_image_gps(image):
    """Read GPS coordinates from EXIF metadata, if present."""
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
        timeout=15
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
            "timezone": "auto"
        },
        timeout=20
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
        "timezone": data.get("timezone", "Local time")
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


st.title("🔥 FireGridNet")
st.subheader("AI-Powered Wildfire Detection & Risk Assessment")
st.write(
    "Detect visible fire and smoke, retrieve weather for a selected "
    "location, and calculate an experimental risk estimate."
)
st.info(
    "Weather comes from a weather model for the selected coordinates. "
    "It is not measured directly from the photograph or necessarily "
    "at the exact forest site."
)

st.divider()

left, right = st.columns(2)

with left:
    st.header("1. Fire & Smoke Detection")

    uploaded_image = st.file_uploader(
        "Upload a forest or wildfire image",
        type=["jpg", "jpeg", "png"]
    )

    image_gps = None

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_gps = get_image_gps(image)
        image = image.convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if image_gps:
            st.success(
                f"Image GPS found: {image_gps[0]:.5f}, "
                f"{image_gps[1]:.5f}"
            )
        else:
            st.caption(
                "No readable GPS coordinates found in image metadata."
            )

        if st.button("Analyze Image"):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp:
                    image.save(temp.name)
                    temp_path = temp.name

                result = model.predict(
                    source=temp_path,
                    conf=0.25,
                    verbose=False
                )[0]

                plotted = result.plot()[:, :, ::-1]
                st.image(
                    plotted,
                    caption="AI detection results",
                    use_container_width=True
                )

                if len(result.boxes) == 0:
                    st.warning(
                        "No fire or smoke detected above the threshold. "
                        "This does not prove the area is safe."
                    )
                else:
                    st.success(
                        f"Detected {len(result.boxes)} possible "
                        "fire/smoke object(s)."
                    )

                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        st.write(
                            f"**{model.names[class_id].title()}** — "
                            f"{confidence:.1%} confidence"
                        )

            except Exception as error:
                st.error(f"Image analysis failed: {error}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

with right:
    st.header("2. Location & Weather")

    location_mode = st.radio(
        "Choose location method",
        [
            "Image GPS",
            "Search for a place",
            "Enter map coordinates"
        ],
        horizontal=False
    )

    selected_location = None
    lat = None
    lon = None

    if location_mode == "Image GPS":
        if image_gps:
            lat, lon = image_gps
            selected_location = "Coordinates from image metadata"
            st.write(f"Coordinates: {lat:.5f}, {lon:.5f}")
        else:
            st.warning(
                "This image has no readable GPS data. Choose place "
                "search or enter coordinates instead."
            )

    elif location_mode == "Search for a place":
        place = st.text_input(
            "Forest, national park, town, or place name"
        )

        if st.button("Find Location"):
            if place.strip():
                try:
                    places = find_location(place.strip())
                    if places:
                        st.session_state["places"] = places
                    else:
                        st.warning("No matching locations found.")
                except requests.RequestException as error:
                    st.error(f"Location search failed: {error}")

        places = st.session_state.get("places", [])

        if places:
            choices = [
                (
                    f"{p['name']}, {p.get('admin1', '')}, "
                    f"{p.get('country', '')} "
                    f"({p['latitude']:.3f}, {p['longitude']:.3f})"
                )
                for p in places
            ]
            chosen = st.selectbox("Choose the matching location", choices)
            selected_index = choices.index(chosen)
            selected_place = places[selected_index]
            lat = selected_place["latitude"]
            lon = selected_place["longitude"]
            selected_location = chosen

    else:
        lat = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            value=20.0, format="%.5f"
        )
        lon = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            value=78.0, format="%.5f"
        )
        selected_location = "Manually entered coordinates"

    weather_mode = st.radio(
        "Weather data source",
        ["Fetch weather automatically", "Enter weather manually"]
    )

    if weather_mode == "Fetch weather automatically":
        if st.button("Fetch Current Weather"):
            if lat is None or lon is None:
                st.warning("Select a valid location first.")
            else:
                try:
                    weather = get_weather(lat, lon)
                    st.session_state["weather"] = weather
                    st.session_state["weather_location"] = (
                        selected_location or f"{lat}, {lon}"
                    )
                    st.session_state["weather_coords"] = (lat, lon)
                except (requests.RequestException, KeyError, ValueError) as error:
                    st.error(f"Could not retrieve weather: {error}")

        weather = st.session_state.get("weather")
        saved_coords = st.session_state.get("weather_coords")
        current_coords = (lat, lon) if lat is not None and lon is not None else None

        if weather and saved_coords == current_coords:
            st.success(
                f"Weather for {st.session_state.get('weather_location', 'selected location')}"
            )
            st.caption(
                f"Observation/model time: {weather['time']} "
                f"({weather['timezone']})"
            )

            a, b = st.columns(2)
            a.metric("Temperature", f"{weather['temperature']} °C")
            b.metric("Humidity", f"{weather['humidity']}%")
            a.metric("Wind speed", f"{weather['wind']} km/h")
            b.metric("Current-hour precipitation", f"{weather['rain_now']} mm")
            st.metric("Precipitation in returned 24-hour window", f"{weather['rain_24h']:.1f} mm")

            if st.button("Calculate Risk From Weather"):
                score, category = calculate_risk(
                    weather["temperature"],
                    weather["humidity"],
                    weather["rain_24h"],
                    weather["wind"]
                )
                st.metric("Experimental risk score", f"{score:.1f}/100")
                st.progress(int(score))
                st.write(f"**Prototype category:** {category} risk")

    else:
        temperature = st.number_input(
            "Temperature (°C)", -10.0, 60.0, 35.0
        )
        humidity = st.slider("Relative humidity (%)", 0, 100, 30)
        rainfall = st.number_input(
            "Rainfall in last 24 hours (mm)",
            0.0, 500.0, 0.0
        )
        wind = st.number_input(
            "Wind speed (km/h)", 0.0, 200.0, 20.0
        )

        if st.button("Calculate Manual Risk"):
            score, category = calculate_risk(
                temperature, humidity, rainfall, wind
            )
            st.metric("Experimental risk score", f"{score:.1f}/100")
            st.progress(int(score))
            st.write(f"**Prototype category:** {category} risk")

st.divider()
st.caption(
    "Academic prototype only. Weather values are model-based estimates "
    "and may not represent conditions at the exact forest location. "
    "The risk formula is illustrative and not scientifically validated. "
    "Do not use it for emergency decisions."
)
