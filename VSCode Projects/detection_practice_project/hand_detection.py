import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


hand_array = np.array([])
cords_list_array = np.array([])
all_cords = []


def get_all_cords(cap_device=0):
    label = None
    all_cords = []
    cap = cv2.VideoCapture(cap_device)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            key = cv2.waitKey(10) & 0xFF
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
            
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
                        print(f'Landmark {num}: ({normalized_x}, {normalized_y}, {normalized_z}, {label})')

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