# Exercise-Monitor
Exercise Monitoring App utilizing MediaPipe &amp; OpenCV from webcam integrated with Streamlit for front end hosting.

~~ Setup & Installation Instructions :
- Install all the files in repo
- For Windows(Powershell or CMD) in terminal :
python -m venv .venv && .venv\Scripts\activate
- For MacOS/Linux in terminal :
python3 -m venv .venv && source .venv/bin/activate
- Also in terminal, Run "pip install cv2 mediapipe numpy subprocess time pygame streamlit os sys"
- Ensure .venv virtual environment is running
- To run, type "streamlit run FinalApp.py" in terminal under .venv environment

~~ Feature Explanation :

Novara Fitness Tracking App is a lightweight computer vision–based fitness assistant that helps users perform exercises with proper form and track their progress in real time. Built using OpenCV, Mediapipe, and Streamlit, it leverages webcam input to detect body landmarks, analyze posture, and count repetitions automatically.

- Key Features

Real-Time Exercise Tracking
Continuously monitors the user’s movement through live camera input, enabling instant feedback on posture and progress.

Pose Detection Using Mediapipe
Employs Mediapipe’s pose estimation model to identify key body landmarks with high accuracy, forming the foundation for angle and movement analysis.

Automated Rep Counting
Calculates joint angles and detects motion phases to count repetitions for exercises such as Bicep Curls and Lateral Raises with reliable accuracy.

Posture Validation
Provides visual and auditory cues to ensure users maintain proper form, helping prevent incorrect movements and reduce injury risk.

Lightweight and Hardware-Free
Runs directly in the browser through Streamlit, requiring only a standard webcam—no additional devices or sensors needed.

Flexible and Portable
Designed to work seamlessly in various environments—whether at home, in the gym, or in a workspace—making it accessible for all fitness levels.

~~ Dependencies & Requirements :
- Refer to requirements.txt & install streamlit==1.50.0 ; opencv-python==4.11.0.86 ; mediapipe==0.10.21 ; numpy==1.26.4 ; pygame==2.6.1
- install cv2 mediapipe numpy subprocess time pygame streamlit os sys
- Ensure to have python version compatible with mediapipe (Python 3.11.0 is recommended)
- Use .venv virtual environment
