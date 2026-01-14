import joblib
import os

MODEL_PATH = os.path.join("models", "best_model.joblib")

model = joblib.load(MODEL_PATH)

# Three spam examples from dataset
spam_texts = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    "WINNER!! As a valued network customer you have been selected to receivea 900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "URGENT! You have won a 1 week FREE membership in our 100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"
]

for text in spam_texts:
    prediction = model.predict([text])[0]
    print(f"Text: {text[:50]}...")
    print(f"Prediction: {prediction}")
    print("---")
