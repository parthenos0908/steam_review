from sklearn.metrics import accuracy_score, classification_report

y_test = [0,0,0,1,1,1,2,2,2,3,3,3]
y_pred = [0,1,0,3,1,1,2,2,2,0,3,3]

target_names = ['Bug', 'Rating', 'Feature', 'UserExperience']

print(classification_report(y_test, y_pred))