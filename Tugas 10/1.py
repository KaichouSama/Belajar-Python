# import json
# import random
# from deep_translator import GoogleTranslator

# # Baca intents.json
# with open(r'C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 10/intents.json', 'r', encoding='utf-8') as file:
#     intents_data = json.load(file)


# # Fungsi translate pakai deep-translator
# def translate_text(text, source_lang, target_lang):
#     return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

# # Prediksi intent sederhana (pattern matching manual)
# def predict_intent(user_input):
#     for intent in intents_data['intents']:
#         for pattern in intent['patterns']:
#             if pattern.lower() in user_input.lower():
#                 return intent['tag']
#     return "noanswer"

# # Fungsi chatbot
# def chatbot_response(user_input):
#     # Translate ID -> EN
#     translated_input = translate_text(user_input, 'id', 'en')
#     print(f"User (ID): {user_input} -> Translated to EN: {translated_input}")

#     # Prediksi intent
#     intent_tag = predict_intent(translated_input)

#     if intent_tag != "noanswer":
#         for intent in intents_data['intents']:
#             if intent['tag'] == intent_tag:
#                 response_en = random.choice(intent['responses'])
#                 break
#     else:
#         response_en = "Sorry, I don't understand."

#     # Translate EN -> ID
#     response_id = translate_text(response_en, 'en', 'id')
#     return response_id

# # === Simulasi User ===
# while True:
#     user_input = input("Kamu: ")
#     if user_input.lower() in ['exit', 'keluar', 'quit']:
#         print("Chatbot: Sampai jumpa!")
#         break
#     response = chatbot_response(user_input)
#     print(f"Chatbot: {response}")


import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Baca intents.json Bahasa Indonesia
with open(r'C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 10/intents_bi.json', 'r', encoding='utf-8') as file:
    intents_data = json.load(file)

# Persiapkan data untuk pelatihan
patterns = []
tags = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Tokenisasi dan vektorisasi menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)  # Representasi numerik dari input
y = np.array(tags)  # Label target (intent)

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")

# Prediksi intent dari input pengguna
def predict_intent(user_input):
    # Transformasi input pengguna menjadi representasi numerik menggunakan TF-IDF Vectorizer
    input_vect = vectorizer.transform([user_input])
    
    # Prediksi intent menggunakan model Naive Bayes
    predicted_tag = model.predict(input_vect)[0]
    
    return predicted_tag

# Respon chatbot berdasarkan intent yang diprediksi
def chatbot_response(user_input):
    intent_tag = predict_intent(user_input)
    
    for intent in intents_data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    
    return "Maaf, saya tidak mengerti."

# Simulasi Chatbot
while True:
    user_input = input("Kamu: ")
    if user_input.lower() in ['exit', 'keluar', 'quit']:
        print("Chatbot: Sampai jumpa!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")

