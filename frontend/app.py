import streamlit as st
import requests
import base64
import time
from PIL import Image
import io

st.set_page_config(page_title="AI CCTV Surveillance", layout="wide")
st.title("🛡️ AI CCTV Surveillance System")

video = st.text_input("Enter CCTV video path:")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Monitoring"):
        requests.post("http://localhost:8000/start", params={"video_path": video})
        st.session_state.running = True

with col2:
    if st.button("⏹ Stop"):
        st.session_state.running = False

video_box = st.empty()
status_box = st.empty()
alert_box = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False

while st.session_state.running:
    try:
        r = requests.get("http://localhost:8000/frame").json()

        if r["status"] == "ended":
            st.warning("Video finished")
            break

        frame_bytes = base64.b64decode(r["frame"])
        img = Image.open(io.BytesIO(frame_bytes))

        video_box.image(img, use_column_width=True)

        status_box.markdown(f"### Status: **{r['label']}**")

        if r["label"] == "VIOLENCE":
            alert_box.error("🚨 VIOLENCE DETECTED")
        else:
            alert_box.success("✅ Normal Activity")

        time.sleep(0.03)

    except:
        break
