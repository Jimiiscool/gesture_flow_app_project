from sklearn.model_selection import train_test_split
import numpy as np

new_data = np.load('C:/Users/ajibo/github_repos/gesture_flow_app_project/data/practice_project_data/hand_cords.npy')

X = new_data[:, :-1]
y = new_data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
                )

print(X_train.shape)
print(X_test.shape)

print(set(y_train))
print(set(y_test))
