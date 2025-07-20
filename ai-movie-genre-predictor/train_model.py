import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 🧼 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 📦 Load and preprocess data
df = pd.read_csv("train_data.txt", sep=" ::: ", names=["id", "title", "genre", "description"], engine='python')
df.dropna(subset=["description", "genre"], inplace=True)

# Convert multi-label genres to single by selecting first genre
df['genre'] = df['genre'].str.split('|').str[0]

# Balance genres: limit to top 5 genres with max 1000 samples each
top_genres = df['genre'].value_counts().nlargest(5).index.tolist()
df = df[df['genre'].isin(top_genres)]
df = df.groupby('genre').apply(lambda x: x.sample(min(len(x), 1000), random_state=42)).reset_index(drop=True)

# Clean descriptions
df['description'] = df['description'].apply(clean_text)

# 🚂 Split data
X_train, X_test, y_train, y_test = train_test_split(df['description'], df['genre'], test_size=0.2, random_state=42)

# 🧠 Build and train model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# 💾 Save model
joblib.dump(model, "genre_model.pkl")
