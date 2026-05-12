import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from model_train import y_pred, y_test, X_test
# Load the trained model

test_model = joblib.load('C:/Users/ajibo/github_repos/gesture_flow_app_project/models/practice_project_models/first_model.pkl')


y_pred = test_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')