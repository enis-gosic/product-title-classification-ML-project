import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib


# 1️⃣ Load dataset
url = "https://raw.githubusercontent.com/enis-gosic/product-title-classification-ML-project/main/data/products.csv"
df = pd.read_csv(url)

# 2️⃣ Clean column names
df.columns = df.columns.str.strip().str.lstrip('_').str.lower().str.replace(' ', '_')

# 3️⃣ Drop rows with missing essential values
df = df.dropna(subset=['product_title', 'category_label'])

# 4️⃣ Standardize category labels
category_map = {
    'Mobile Phone': 'Mobile Phones',
    'fridge': 'Fridges',
    'CPU': 'CPUs'
}

df['category_label'] = df['category_label'].replace(category_map)

# 5️⃣ Feature engineering on product_title
df['title_char_count'] = df['product_title'].astype(str).apply(len)
df['title_word_count'] = df['product_title'].astype(str).apply(lambda x: len(x.split()))
df['title_has_numbers'] = df['product_title'].astype(str).str.contains(r'\d').astype(int)
df['title_has_special_chars'] = df['product_title'].astype(str).str.contains(r'[^a-zA-Z0-9 ]').astype(int)
df['title_uppercase_count'] = df['product_title'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))
df['title_longest_word_length'] = df['product_title'].astype(str).apply(lambda x: max([len(w) for w in x.split()]) if x.split() else 0)

# 6️⃣ Define features and target
numerical_features = [
    'title_char_count',
    'title_word_count',
    'title_has_numbers',
    'title_has_special_chars',
    'title_uppercase_count',
    'title_longest_word_length'
]

X = df[['product_title'] + numerical_features]
y = df['category_label']

# 7️⃣ Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('title', TfidfVectorizer(), 'product_title'),
        ('numeric', MinMaxScaler(), numerical_features)
    ]
)

# 8️⃣ Train SVM model
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LinearSVC())
])

pipeline.fit(X, y)

# 9️⃣ Save trained model
joblib.dump(pipeline, "model/product_category_svm.pkl")
print("✅ SVM model trained and saved as 'model/product_category_svm.pkl'")
