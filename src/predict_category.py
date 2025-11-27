import joblib
import pandas as pd


# Load trained model
model = joblib.load("model/product_category_svm.pkl")

print("✅ Model loaded successfully!")
print("Type 'exit' at any time to quit.\n")

while True:
    title = input("📝 Enter product title: ")

    if title.lower().strip() == "exit":
        print("Exiting...")
        break

    # Feature engineering (same as in training)
    title_char_count = len(title)
    title_word_count = len(title.split())
    title_has_numbers = 1 if any(char.isdigit() for char in title) else 0
    title_has_special_chars = 1 if any(not char.isalnum() and char != " " for char in title) else 0
    title_uppercase_count = sum(1 for char in title if char.isupper())
    title_longest_word_length = (
        max(len(word) for word in title.split()) if title.split() else 0
    )

    # Create DataFrame for prediction
    user_input = pd.DataFrame([{
        "product_title": title,
        "title_char_count": title_char_count,
        "title_word_count": title_word_count,
        "title_has_numbers": title_has_numbers,
        "title_has_special_chars": title_has_special_chars,
        "title_uppercase_count": title_uppercase_count,
        "title_longest_word_length": title_longest_word_length
    }])

    # Predict category
    prediction = model.predict(user_input)[0]

    print(f"🔎 Predicted Category: **{prediction}**")
    print("-" * 40 + "\n")
