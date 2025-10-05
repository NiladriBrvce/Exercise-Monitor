import cv2
import mediapipe as mp
import numpy as np
import subprocess
import time
import pygame
import streamlit as st
import os
import sys

#Variables for drawing utilities & pose estimation model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Sound Alert
pygame.mixer.init()
sound = pygame.mixer.Sound('boop.mp3')
sound.play()

#Raise counter variables
switch_start = None
switch_duration = 5

#ANGLE CALCULATION FUNCTION
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return angle

#VIDEO FEED - CONSISTENT CODE BLOCK
cap = cv2.VideoCapture(0)

#Raise counter variables
rcounter = 0
lcounter = 0
rstage = None
lstage = None

#Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
        ret, frame = cap.read()

        #Recolor Image TO RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False 

        #Make Detection
        results = pose.process(image) 

        #Recolor Image TO BGR 
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Render Raise Panel
        cv2.rectangle(image, (0,0), (640,50), (0,0,0), -1) # image, start pos, end pos, color, fill box

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #Get Coordinates ~~ x and y coordinates for ear, shoulder & hip stored in respected arrays ear, shoulder & hip
            rsho = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            relb = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            rwrst = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            lsho = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lelb = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            lwrst = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            rwx1, rwy1 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            lwx2, lwy2 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    
            # Calculate Euclidean distance
            dist = np.sqrt((lwx2 - rwx1)**2 + (lwy2 - rwy1)**2)

            if rwx1 is None or rwy1 is None or lwx2 is None or lwy2 is None:
                continue

            #Calculate Angle
            rangle = calculate_angle(rsho, relb, rwrst)
            langle = calculate_angle(lsho, lelb, lwrst)

            #Visualize Angle
            cv2.putText(image, str(rangle), 
                            tuple(np.multiply(relb, [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA 
                                )  

            cv2.putText(image, str(langle), 
                            tuple(np.multiply(lelb, [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA 
                                )     
            
            #Visualize Distance
            cv2.putText(image, str(dist), 
                            tuple(np.multiply(rwrst, [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA 
                                )
            
            #Raise Counter logic
            if rangle > 160:
                rstage = "down"
            if rangle < 50 and rstage == 'down':
                rstage = "up"
                rcounter += 1

            if langle > 160:
                lstage = "down"
            if langle < 50 and lstage == 'down':
                lstage = "up"
                lcounter += 1

            #Switch Exercise logic
            if dist > 0.35:
                switch_stage = "nil"
                switch_start = None
            elif dist <= 0.35 and switch_stage == "nil":
                switch_stage = "switch"
                switch_start = time.time()
            if switch_stage == "switch" and switch_start is not None:
                selapsed = time.time() - switch_start
                sremaining = max(0, int(switch_duration - selapsed))
                if sremaining < 3:
                    cv2.putText(image, f"HOLD FOR: {sremaining+1}", (210, 37),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if selapsed >= switch_duration:
                    cap.release() 
                    cv2.destroyAllWindows()
                    # FinalApp.exercise_page("Lateral Raises", "LateralRaise")
                    os.execv(sys.executable, ['python'] + ['LateralRaise.py'])
                    switch_start = None
            
        except: 
            pass 

        #Rep Data
        cv2.putText(image, 'RIGHT ARM REPS', (20,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(rcounter),
                    (18,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'LEFT ARM REPS', (500,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(lcounter),
                    (500,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        #Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) 

        cv2.imshow('Mediapipe Feed', image) 

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    cap.release() 
    cv2.destroyAllWindows()