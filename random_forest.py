import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset.csv')
x = df[['age', 'position', 'history_injuries_2_seasons', 'minutes_last_week', 'minutes_prior_3w', 'sprints_last_week', 'games_last_14d', 'days_since_last_injury', 'rest_days_until_next_game']]
y = df['y']

x['position'] = x['position'].map({'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

sample = x_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()

print(f"\nSample Player: {sample_dict}")
print(f"Predicted Injury: {'Injury' if prediction[0] == 1 else 'No Injury'}")

