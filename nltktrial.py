import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

human_text = [
    "This is a human-written text.",
    "I believe in human creativity.",
    "Humans have unique insights.",
]

ai_text = [
    "I am an AI language model.",
    "Artificial intelligence is fascinating.",
    "AI can generate realistic text.",
]

text_data = human_text + ai_text
target = ['human'] * len(human_text) + ['ai'] * len(ai_text)

vectorizer = CountVectorizer()
feature_vectors = vectorizer.fit_transform(text_data)

classifier = MultinomialNB()
classifier.fit(feature_vectors, target)

def detect_ai(text):
    input_vector = vectorizer.transform([text])
    probabilities = classifier.predict_proba(input_vector)[0]
    ai_probability = probabilities[1]
    human_probability = probabilities[0]
    total_probability = ai_probability + human_probability
    ai_percentage = (ai_probability / total_probability) * 100
    return ai_percentage

input_text = "In the midst of a bustling city, where towering skyscrapers kissed the clouds and streams of people hurriedly rushed through the crowded streets, a young artist sat on a weathered bench, their sketchbook resting on their lap, capturing the vibrant energy and essence of the urban landscape with each stroke of their pencil."
percentage = detect_ai(input_text)

print(f"Ai mark is {percentage:.2f}%")
