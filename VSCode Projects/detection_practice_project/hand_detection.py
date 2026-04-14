import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cords_list_array = np.array([]).reshape(-1, 3)



def count_fingers():
    fingers_up = []
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
    return result


cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_height, img_width, img_channel = img.shape

        if results.multi_hand_landmarks:
            num_fingers = []
            cords_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                result = count_fingers()
                for num, cords in enumerate(hand_landmarks.landmark):
                    x_cords = f'X Cords: {cords.x}'
                    y_cords = f'Y Cords: {cords.y}'
                    normalized_x = wrist.x - cords.x
                    normalized_y = wrist.y - cords.y
                    normalized_z = wrist.z - cords.z
                    hand_array = np.array([[normalized_x, normalized_y, normalized_z]]).flatten()
                    cords_list.append((x_cords, y_cords))
                    num_fingers.append(result)
                    print(result)
                    cords_list_array = np.vstack([cords_list_array, hand_array])
                    

                    
        cv2.imshow('Hand Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
print(cords_list_array)
print(sum(num_fingers))        
cap.release()
cv2.destroyAllWindows()