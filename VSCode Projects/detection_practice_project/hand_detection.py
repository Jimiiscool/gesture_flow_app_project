import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

fingers_up = []
#num_fingers = []
#cords_list = []
hand_array = np.array([])
cords_list_array = np.array([])
all_cords = []




'''def count_fingers():
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_pip = hand_landmarks.landmark[6]
            if index_finger_tip.y < index_finger_pip.y:
                    fingers_up.append(1)
            elif index_finger_tip.y > index_finger_pip.y or len(fingers_up) >= 5:
                    fingers_up.append(0)
            else:
                    fingers_up.append(0)
            result = sum(fingers_up)
    return result'''

def get_cords():
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
            #img_height, img_width, img_channel = img.shape


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
                    all_cords.append(frame_cords)
                    print(frame_cords)

                        

                        
            cv2.imshow('Hand Detection', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cords_list_array = np.array(all_cords)
        print(cords_list_array.shape)
        final = cords_list_array
        cap.release()
        cv2.destroyAllWindows()
        return final
    

get_cords()