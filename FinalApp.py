import streamlit as st
import subprocess
import pygame

def exercise_page(title, script_key):
    st.header(title)
    st.write("Press Q to exit Session")
    st.markdown("---")  # Separator line
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.session_state[f"{script_key}_started"] = False
        return 

    if not st.session_state.get(f"{script_key}_started", False):
        st.session_state[f"{script_key}_started"] = True
        subprocess.run(["python", f"{script_key}.py"])

def home():
    # Title and description
    st.markdown(
        """
        <h1 style="text-align:center; color:#4CAF50;">
            🍂 Novare Fitness App 🍂
        </h1>
        <p style="text-align:center; font-size:18px; color:#555;">
            Select an exercise below and track your performance in real-time using AI-powered pose detection.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Centering buttons with columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.empty()

    with col2:
        # Add exercise images
        st.image("bicep.png", use_container_width=True)
        if st.button("Start Bicep Curls 🏋️"):
            st.session_state.page = "bicep_curls"
            st.session_state["BicepCurl_started"] = False
        st.image("lateral.png", use_container_width=True)
        if st.button("Start Lateral Raises 💪"):
            st.session_state.page = "lateral_raises"
            st.session_state["LateralRaise_started"] = False

    with col3:
        st.empty()

    # Footer
    st.markdown(
        """
        <hr>
        <p style="text-align:center; color:gray; font-size:14px;">
            Powered by OpenCV + Mediapipe | Built with ❤️ using Streamlit
        </p>
        """,
        unsafe_allow_html=True,
    )

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Page routing
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "bicep_curls":
        exercise_page("Bicep Curls", "BicepCurl")
        #subprocess.run(["python", "BicepCurl.py"])
    elif st.session_state.page == "lateral_raises":
        exercise_page("Lateral Raises", "LateralRaise")
        #subprocess.run(["python", "LateralRaise.py"])

if __name__ == "__main__":
    main()
