# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
import string 
import tkinter as tk
from tkinter import simpledialog

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence = 0.3)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('model.tf')

# Load class names
alphabet = list(string.ascii_uppercase) + [' ', '.']
print(alphabet)

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# initializing some useful values 
i = 0
generated_input = ''
usr_input = ''

# initializing input window 
ROOT = tk.Tk()

ROOT.withdraw()
USER_INP = simpledialog.askstring(title="HEY!", prompt="TYPE YOUR WORD HERE (IN CAPS LOCK)")
usr_input = USER_INP
input_size = cv2.getTextSize(usr_input, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
input_X = int((1280 - input_size[0]) / 2)

print('input:', input[3])

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    
    className = ''

    cv2.putText(frame, input, (input_X, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = alphabet[classID - 1]
    
    if className == ' ':
        cv2.putText(frame, 'Your letter: SPACE', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Your letter:{}'.format(className), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.putText(frame, generated_input, (input_X, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    if i == len(input):
        outcome = 'You did a great job!'
        outcome_size = cv2.getTextSize(outcome, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)[0]
        outcome_X = int((1280 - outcome_size[0]) / 2)
        cv2.putText(frame, outcome, (outcome_X, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)
    elif className == input[i]:
        if className != ' ':
            generated_input = generated_input + input[i]
            i = i+1
        else:
            generated_input = generated_input + ' '
            i = i+1


    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()