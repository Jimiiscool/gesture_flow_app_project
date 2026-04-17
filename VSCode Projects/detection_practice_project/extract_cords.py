import numpy as np
import csv
from hand_detection import get_all_cords


data = get_all_cords(0)

np.save('hand_cords.npy', data)

with open('C:/Users/ajibo/github_repos/gesture_flow_app_project/data/practice_project_data/hand_cords.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

    print('data.shape')