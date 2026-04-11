'''from hand_detection import results
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
    return result'''