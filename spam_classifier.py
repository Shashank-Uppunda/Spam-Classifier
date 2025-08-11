# Email/SMS Spam Classifier using Naive Bayes (scikit-learn)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Only keep 'label' (ham/spam) and 'message' columns
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename columns

# Convert labels to numbers: spam=1, ham=0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Convert text into numbers using Bag-of-Words model
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)  # Learn vocabulary + transform training data
X_test_counts = vectorizer.transform(X_test)        # Transform test data using same vocabulary

# Create and train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Test model on test data
y_pred = model.predict(X_test_counts)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Allow user to enter messages for prediction
print("\nType a message to classify as Spam or Ham.")
print("Enter 'X' to stop.\n")

while True:
    user_msg = input("Enter message: ")
    if user_msg.strip().upper() == 'X':  # Exit condition
        print("Stopping prediction. Goodbye!")
        break

    # Convert user message to numerical form
    user_msg_counts = vectorizer.transform([user_msg])
    prediction = model.predict(user_msg_counts)[0]

    print("Prediction:", "Spam" if prediction == 1 else "Ham")
    print("-" * 40)
