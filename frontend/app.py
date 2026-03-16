import streamlit as st
import json
import hashlib
import os
import requests
import time
import base64
from PIL import Image
import io
import pandas as pd

# ================= STREAMLIT CONFIG =================

st.set_page_config(page_title="AI CCTV System", layout="wide")

# ================= USER DATABASE =================

USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

def login_user(username, password):
    users = load_users()
    return users.get(username) == hash_password(password)

# ================= SESSION STATE =================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "video_path" not in st.session_state:
    st.session_state.video_path = ""

if "running" not in st.session_state:
    st.session_state.running = False

if "crowd_data" not in st.session_state:
    st.session_state.crowd_data = []

# ================= LOGIN PAGE =================

if not st.session_state.logged_in:

    st.title("🔐 AI CCTV Surveillance Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if login_user(username, password):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.success("Login successful!")
                st.rerun()

            else:
                st.error("Invalid username or password")

    with tab2:

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):

            if register_user(new_user, new_pass):

                st.success("User registered successfully! Now login.")

            else:
                st.error("User already exists!")

# ================= DASHBOARD =================

else:

    st.title(f"🛡️ Welcome {st.session_state.username}")
    st.subheader("AI CCTV Surveillance Dashboard")

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.running = False
        st.rerun()

    st.markdown("---")

    # ================= VIDEO SOURCE =================

    st.header("📹 Select Video Source")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Upload Video")

        uploaded_file = st.file_uploader("Upload CCTV Video", type=["mp4","avi"])

        if uploaded_file:

            save_path = f"uploaded_{uploaded_file.name}"

            with open(save_path,"wb") as f:
                f.write(uploaded_file.read())

            st.session_state.video_path = save_path

            st.success(f"Uploaded: {save_path}")

    with col2:

        st.subheader("Existing CCTV Feeds")

        if st.button("🏬 Mall CCTV"):
            st.session_state.video_path = "data/demo_videos/mall.mp4"

        if st.button("🏫 Campus CCTV"):
            st.session_state.video_path = "data/demo_videos/campus.mp4"

        if st.button("🚦 Street CCTV"):
            st.session_state.video_path = "data/demo_videos/street.mp4"

    with col3:

        st.subheader("Live Camera")

        if st.button("🎥 Use Webcam"):
            st.session_state.video_path = "webcam"

    st.markdown("---")

    st.write(f"**Selected Source:** {st.session_state.video_path}")

    # ================= START STOP =================

    colA, colB = st.columns(2)

    with colA:

        if st.button("▶ Start Monitoring"):

            if st.session_state.video_path == "":

                st.error("Select a video source first!")

            else:

                requests.post(
                    "http://localhost:8000/start",
                    params={"video_path": st.session_state.video_path}
                )

                st.session_state.running = True
                st.success("Monitoring Started!")

    with colB:

        if st.button("⏹ Stop Monitoring"):

            st.session_state.running = False
            st.warning("Monitoring Stopped")

    st.markdown("---")

    # ================= SPLIT SCREEN =================

    left_col, right_col = st.columns([2,1])

    with left_col:

        st.subheader("📺 Live CCTV Feed")

        video_box = st.empty()

    with right_col:

        st.subheader("📊 AI Surveillance Dashboard")

        people_metric = st.metric("People Count", "0")

        threat_metric = st.metric("Threat Level", "NORMAL")

        st.write("### Crowd Density")

        chart_placeholder = st.empty()

    # ================= STREAM LOOP =================

    while st.session_state.running:

        try:

            r = requests.get("http://localhost:8000/frame").json()

            if r["status"] != "ok":

                time.sleep(0.3)
                continue

            frame_bytes = base64.b64decode(r["frame"])

            img = Image.open(io.BytesIO(frame_bytes))

            video_box.image(img, use_container_width=True)

            # -------- THREAT STATUS --------

            label = r.get("label","NORMAL")

            confidence = r.get("confidence",0)

            threat_metric.metric("Threat Level", label)

            # -------- PEOPLE COUNT (estimate) --------

            people_count = r.get("person_count",0)

            people_metric.metric("People Count", people_count)

            # -------- CROWD GRAPH --------

            st.session_state.crowd_data.append(people_count)

            if len(st.session_state.crowd_data) > 40:
                st.session_state.crowd_data.pop(0)

            chart_placeholder.line_chart(
                pd.DataFrame(st.session_state.crowd_data)
            )

            time.sleep(0.08)

        except Exception as e:

            st.error(f"Stream stopped: {e}")
            break