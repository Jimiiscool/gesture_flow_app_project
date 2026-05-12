import mediapipe as mp
import cv2
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import joblib
from collections import deque
from collections import defaultdict

def get_smoothed_prediction(buffer, decay=0.9):
    scores = defaultdict(float)

    for i, (pred, conf) in enumerate(buffer):
         time_weight = decay**(len(buffer)-i) # Give more weight to recent predictions
         scores[pred] += conf * time_weight
        
    return max(scores, key=scores.get)

def raw_predict(features, model):
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]  # Get the confidence of the predicted class
    
    class_index = list(model.classes_).index(prediction)  # Get the index of the predicted class
    confidence = confidence[class_index]

pred_buffer = deque(maxlen = 7)  # Buffer to hold the last 6 predictions


test_model = joblib.load('C:/Users/ajibo/github_repos/gesture_flow_app_project/models/practice_project_models/first_model.pkl') 


hand_array = np.array([])
cords_list_array = np.array([])
all_cords = []
label_map = ['Fist', 'Open', 'Peace']  # Map numeric predictions to gesture labels

label = None
all_cords = []
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  


            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist = hand_landmarks.landmark[0]
                    ref = hand_landmarks.landmark[9]
                    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)

                    frame_cords = []
                    for num, cords in enumerate(hand_landmarks.landmark):
                        normalized_x = (cords.x - wrist.x)/scale
                        normalized_y = (cords.y - wrist.y)/scale
                        normalized_z = (cords.z - wrist.z)/scale
                        frame_cords.extend([normalized_x, normalized_y, normalized_z])
                    features = np.array(frame_cords).reshape(1, -1)
                    prediction, confidence = raw_predict(features, test_model)
                    if confidence > 0.60:
                        pred_buffer.append((prediction[0], confidence))
                    else:
                        pass  # Add the current prediction to the buffer
                    smoothed_prediction = get_smoothed_prediction(pred_buffer, decay=0.9)  # Get the smoothed prediction from the buffer
                    cv2.putText(img, f"Prediction: {prediction[0]}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if confidence > 0.50:  
                        cv2.putText(img, f"Confidence: {confidence:.2f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, f"Unsure of Gesture", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else: 
                 cv2.putText(img, f"No Hand Detected", (10, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                            

            cv2.imshow('Hand Detection', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
