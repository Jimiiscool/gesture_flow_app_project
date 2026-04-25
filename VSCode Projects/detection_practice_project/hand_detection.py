import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


hand_array = np.array([])
cords_list_array = np.array([])
all_cords = []
label_counts = {'Fist': 0, 'Open': 0, 'Peace': 0}


def get_all_cords(cap_device=0):
    label = None
    all_cords = []
    cap = cv2.VideoCapture(cap_device)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

            key = cv2.waitKey(10) & 0xFF
  
            if key == ord('1'):
                label = 'Fist'
            elif key == ord('2'):
                label = 'Open'
            elif key == ord('3'):
                label = 'Peace'
            else:
                pass

            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist = hand_landmarks.landmark[0]
                    frame_cords = []
                    for num, cords in enumerate(hand_landmarks.landmark):
                        normalized_x = cords.x - wrist.x
                        normalized_y = cords.y - wrist.y
                        normalized_z = cords.z - wrist.z
                        frame_cords.extend([normalized_x, normalized_y, normalized_z])
                        
                    if label is not None:
                        frame_cords.append(label)
                        all_cords.append(frame_cords)
                        label_counts[label] += 1
                        print(f'Landmark {num}: ({normalized_x}, {normalized_y}, {normalized_z}, {label})')
                        frame_count = len(all_cords)
                        cv2.putText(img, f"Frames: {frame_count}", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.putText(img, f"Fist: {label_counts['Fist']}", (10, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img, f"Open: {label_counts['Open']}", (10, 170),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img, f"Peace: {label_counts['Peace']}", (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    else:
                        pass

                            

            cv2.imshow('Hand Detection', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cords_list_array = np.array(all_cords)
        print(cords_list_array.shape)
        final = cords_list_array
        cap.release()
        cv2.destroyAllWindows()
        return final


get_all_cords(0)