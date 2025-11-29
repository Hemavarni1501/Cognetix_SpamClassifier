import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sys

# --- 1. Load and Preprocess the Dataset ---
file_path = 'spam.csv'
try:
    # Load the data, specifying encoding as it is common with this dataset
    df = pd.read_csv(file_path, encoding='latin-1')
    # Keep only the columns we need and rename them
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    print("Dataset loaded successfully.")
    # Map the labels to binary (0 for ham, 1 for spam)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Ensure it is in the same directory.")
    sys.exit()

# Define the text cleaning function (removes punctuation and stopwords)
def text_process(mess):
    """
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean words
    """
    # Use a set comprehension for faster punctuation removal
    nopunc = "".join([char for char in mess if char not in string.punctuation])
    
    # Use a set for faster lookup of stopwords
    english_stopwords = set(stopwords.words('english'))
    return [word.lower() for word in nopunc.split() if word.lower() not in english_stopwords]

# --- 2. Convert Text Data into Numerical Features using CountVectorizer ---
# Use CountVectorizer with the custom tokenizer to perform all steps at once
vectorizer = CountVectorizer(analyzer=text_process)
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

print(f"\nText vectorized. Total features (unique words): {X.shape[1]}")

# --- 3. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# --- 4. Train a Naive Bayes Classifier (MultinomialNB) ---
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel trained successfully: Multinomial Naive Bayes Classifier.")

# --- 5. & 6. Evaluate Model Performance and Display Reports ---
print("\n--- Model Evaluation Metrics ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy:.4f}")
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix (Visualization)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix for Spam Classifier')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_spam.png')
# plt.show()


# --- 7. Interactive Prediction of New Samples (User Input) ---
print("\n" + "="*50)
print("--- INTERACTIVE SPAM PREDICTOR ---")
print("Enter a message to classify (type 'quit' to exit).")
print("="*50)

while True:
    try:
        # Get user input from the terminal
        user_input = input("Enter message: ")
        
        if user_input.lower() == 'quit':
            print("Exiting predictor. Goodbye!")
            break
        
        if not user_input.strip():
            print("Please enter a valid message.")
            continue

        # 1. Transform the single message using the *fitted* vectorizer
        input_vectorized = vectorizer.transform([user_input])
        
        # 2. Make the prediction
        prediction = model.predict(input_vectorized)[0]
        
        # 3. Display the result
        status = 'SPAM 🚨' if prediction == 1 else 'HAM ✅'
        print(f"-> CLASSIFICATION: {status}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break