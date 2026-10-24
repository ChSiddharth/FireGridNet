import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="FireGridNet",
    page_icon="🔥",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("/content/best.pt")

model = load_model()

st.title("🔥 FireGridNet")
st.subheader("AI-Powered Wildfire Detection & Risk Assessment")
st.write(
    "Upload an image to detect visible fire and smoke. "
    "Weather risk is a prototype estimate, not an official forecast."
)

st.divider()

left, right = st.columns(2)

with left:
    st.header("1. Fire & Smoke Detection")
    uploaded_image = st.file_uploader(
        "Upload a forest or wildfire image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze Image"):
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp:
                image.save(temp.name)
                temp_path = temp.name

            try:
                results = model.predict(
                    source=temp_path,
                    conf=0.25,
                    verbose=False
                )
                result = results[0]
                plotted = result.plot()
                plotted = plotted[:, :, ::-1]

                st.image(
                    plotted,
                    caption="AI detection results",
                    use_container_width=True
                )

                if len(result.boxes) == 0:
                    st.warning(
                        "No fire or smoke detected above the confidence "
                        "threshold. This does not prove the area is safe."
                    )
                else:
                    st.success(
                        f"Detected {len(result.boxes)} possible "
                        "fire/smoke object(s)."
                    )

                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = model.names[class_id]
                        st.write(
                            f"**{class_name.title()}** — "
                            f"confidence: {confidence:.1%}"
                        )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

with right:
    st.header("2. Weather-Based Risk Estimate")
    st.caption("Enter weather values to calculate a simple prototype score.")

    temperature = st.number_input(
        "Temperature (°C)", min_value=-10.0,
        max_value=60.0, value=35.0
    )
    humidity = st.slider("Relative humidity (%)", 0, 100, 30)
    rainfall = st.number_input(
        "Recent rainfall (mm)", min_value=0.0,
        max_value=500.0, value=0.0
    )
    wind = st.number_input(
        "Wind speed (km/h)", min_value=0.0,
        max_value=200.0, value=20.0
    )

    if st.button("Calculate Risk"):
        score = (
            0.35 * (temperature / 60 * 100)
            + 0.30 * ((100 - humidity))
            + 0.20 * (min(wind, 100) / 100 * 100)
            + 0.15 * (100 - min(rainfall, 50) / 50 * 100)
        )
        score = max(0, min(100, score))

        if score < 30:
            category = "Lower"
        elif score < 60:
            category = "Moderate"
        else:
            category = "Higher"

        st.metric("Prototype risk score", f"{score:.1f}/100")
        st.progress(int(score))
        st.write(f"**Prototype category:** {category} risk")
        st.caption(
            "This score uses illustrative weights and is not scientifically "
            "validated. Do not use it for emergency decisions."
        )

st.divider()
st.caption(
    "FireGridNet is an academic prototype. AI detections can be wrong, "
    "and the weather score is not an operational wildfire prediction."
)
