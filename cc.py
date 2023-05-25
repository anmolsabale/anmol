import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the credit card dataset
data = pd.read_csv(r'C:\Users\Windows\Downloads\creditcard.csv')

# Preprocessing
X = data.drop('Class', axis=1)  # Features
y = data['Class']               # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance:")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-Score: {:.2f}%".format(f1 * 100))


# Evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1]

# Create a bar plot
plt.bar(metrics, scores)
plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1
plt.title('Model Performance')
plt.xlabel('Metrics')
plt.ylabel('Scores')

# Add text labels to each bar
for i, score in enumerate(scores):
    plt.text(i, score, "{:.2f}".format(score), ha='center', va='bottom')

plt.show()

fraud_data = data[data['Class'] == 1]
non_fraud_data = data[data['Class'] == 0]

# Create the scatter plot
plt.scatter(fraud_data['V1'], fraud_data['V2'], color='red', label='Fraud')
plt.scatter(non_fraud_data['V1'], non_fraud_data['V2'], color='blue', label='Non-Fraud')

# Set the plot labels and title
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Credit Card Fraud Detection')

# Add legend
plt.legend()

# Display the plot
plt.show()