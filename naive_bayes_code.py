import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("easy_queue_data.csv")

X = data[['wait_time', 'service_time']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)

for i in range(5):
    print("Wait Time:", X_test.iloc[i]['wait_time'])
    print("Service Time:", X_test.iloc[i]['service_time'])
    print("Probability Normal:", probabilities[i][0])
    print("Probability Problematic:", probabilities[i][1])
    print("Predicted Label:", predictions[i])
    print("Actual Label:", y_test.iloc[i])
    print("----------------------------")
