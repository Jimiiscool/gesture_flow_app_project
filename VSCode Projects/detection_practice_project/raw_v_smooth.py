import mediapipe as mp
import cv2
import numpy as np
import math
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import joblib
from collections import deque
from collections import defaultdict
from matplotlib import pyplot as plt

test_model = joblib.load('C:/Users/ajibo/github_repos/gesture_flow_app_project/models/practice_project_models/first_model.pkl')


def raw_predict(features, model):
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]  # Get the confidence of the predicted class
    
    class_index = list(model.classes_).index(prediction)  # Get the index of the predicted class
    confidence = confidence[class_index]
    return prediction, confidence


class smoother:
     def __init__(self, maxlen=7, decay=0.9):
        self.buffer = deque(maxlen=maxlen)
        self.decay = decay
    


     def update_buffer(self, pred, conf):
        if conf > 0.60:
            self.buffer.append((pred, conf))

        if len(self.buffer) == 0:
            return None, 0.0
        
        scores = defaultdict(float)
        total_weight = 0

        for i, (p, c) in enumerate(self.buffer):
            time_weight = self.decay**(len(self.buffer)-i)
            scores[p] += c * time_weight
            total_weight += time_weight
        
        smooth_pred = max(scores, key=scores.get)
        smooth_conf = scores[smooth_pred]/total_weight
        return smooth_pred, smooth_conf
        
        


smoother = smoother()
log = []
frame_id = 0


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
                    smoothed_prediction, smoothed_confidence = smoother.update_buffer(prediction, confidence)  # Get the smoothed prediction from the buffer
                    log.append({
                        'frame': frame_id,
                        'raw_pred': prediction,
                        'raw_conf' : confidence,
                        'smoothed_pred' : smoothed_prediction,
                        'smoothed_conf' : smoothed_confidence,
                    })
                    frame_id += 1
                    
                    cv2.putText(img, f"Prediction: {prediction}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Prediction: {frame_id}", (10, 50),
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

df = pd.DataFrame(log)
df.to_csv('C:/Users/ajibo/github_repos/gesture_flow_app_project/data/practice_project_data/raw_v_smooth.csv')

plt.scatter(df['frame'], df['smoothed_conf'], label = 'Raw Predict', alpha=0.6)
plt.plot(df['frame'], df['smoothed_conf'], color = 'blue', alpha=0.6)
plt.scatter(df['frame'], df['raw_conf'], label = 'Smoothed Predict',  color = 'red', alpha = 0.6,)
plt.plot(df['frame'], df['raw_conf'], color = 'red', alpha = 0.6)
plt.xlabel('Time')
plt.ylabel('Confidence Score')
plt.legend()
plt.xticks(rotation=45)
plt.show()

