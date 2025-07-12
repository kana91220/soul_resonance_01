import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load Data
df = pd.read_csv("roommates_mock_data.csv")

# Step 2: Select Features for Matching
features = df[['Wakeup', 'Sleep', 'Cleanliness', 'IntroExtro', 'StudyTime', 'NoiseTolerance', 'PetFriendly']]

# Step 3: Preprocess
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['Wakeup', 'Sleep', 'StudyTime', 'PetFriendly']),
    ('num', StandardScaler(), ['Cleanliness', 'IntroExtro', 'NoiseTolerance'])
])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X = pipeline.fit_transform(features)

# Step 4: Get similarity matrix
similarity_matrix = cosine_similarity(X)

# Step 5: Define a match function
def find_top_matches(name, top_n=3):
    idx = df[df['Name'] == name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_matches = [
        (df.iloc[i]['Name'], round(score * 100, 2))
        for i, score in similarity_scores[1:top_n+1]
    ]
    
    return top_matches

# Example usage:
print("Top Matches for Kana:")
print(find_top_matches('Kana'))
