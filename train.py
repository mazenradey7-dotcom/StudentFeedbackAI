# ========================================
# Student Feedback Sentiment Analysis
# AI Project - Ready for GitHub / Scholarships
# ========================================

# -------------------------------
# Step 1: Import Libraries
# -------------------------------
print("Step 1: Importing libraries...")
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Step 2: Load Dataset
# -------------------------------
print("\nStep 2: Loading dataset...")
# Dataset CSV must contain: text,label (1=positive, 0=negative)
df = pd.read_csv("data.csv")
print("Dataset loaded successfully!")
print(df.head())

# -------------------------------
# Step 3: Text Cleaning Function
# -------------------------------
print("\nStep 3: Cleaning text...")

# Stopwords list (manual, no NLTK needed)
stop_words = set([
    "a","about","above","after","again","against","all","am","an","and","any","are",
    "as","at","be","because","been","before","being","below","between","both","but",
    "by","could","did","do","does","doing","down","during","each","few","for","from",
    "further","had","has","have","having","he","her","here","hers","herself","him",
    "himself","his","how","i","if","in","into","is","it","its","itself","me","more",
    "most","my","myself","no","nor","not","of","off","on","once","only","or","other",
    "ought","our","ours","ourselves","out","over","own","same","she","should","so",
    "some","such","than","that","the","their","theirs","them","themselves","then",
    "there","these","they","this","those","through","to","too","under","until","up",
    "very","was","we","were","what","when","where","which","while","who","whom","why",
    "with","would","you","your","yours","yourself","yourselves"
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)
print("Text cleaning done!")
print(df[["text","clean_text"]].head())

# -------------------------------
# Step 4: Feature Extraction (TF-IDF)
# -------------------------------
print("\nStep 4: Extracting features with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,2))
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]
print("TF-IDF feature extraction done!")
print("Feature matrix shape:", X.shape)

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
print("\nStep 5: Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------------------------------
# Step 6: Model Training
# -------------------------------
print("\nStep 6: Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Model training completed!")

# -------------------------------
# Step 7: Evaluation
# -------------------------------
print("\nStep 7: Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 8: Test Custom Input
# -------------------------------
print("\nStep 8: Testing custom input...")
def predict_sentiment(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    cleaned = " ".join(words)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    return "Positive" if pred[0] == 1 else "Negative"

test_sentences = [
    "The explanation was very clear and useful",
    "This course is terrible and confusing",
    "I enjoyed the lectures and the content was excellent",
    "I do not understand anything in this subject"
]

for s in test_sentences:
    print(f"Input: {s} -> Sentiment: {predict_sentiment(s)}")

print("\nProject ready for GitHub / Scholarships!")
