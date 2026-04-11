import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


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
                result = count_fingers()
                for num, cords in enumerate(hand_landmarks.landmark):
                    hand_array = np.array([[cords.x, cords.y, cords.z]])
                    x_cords = f'X Cords: {cords.x}'
                    y_cords = f'Y Cords: {cords.y}'
                    cords_list.append((x_cords, y_cords))
                    num_fingers.append(result)
                    print(result)
                    
        cv2.imshow('Hand Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
print(sum(num_fingers))        
cap.release()
cv2.destroyAllWindows()