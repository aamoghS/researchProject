import torch
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
def detect_ai(text):
    inputs = tokenizer.encode_plus(text, padding='longest', truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    ai_probability = probabilities[1].item()
    human_probability = probabilities[0].item()
    total_probability = ai_probability + human_probability
    ai_percentage = (ai_probability / total_probability) * 100
    return ai_percentage
for i in range(100):
    input_text = "deteriorating, this word is 0%"
    percentage = detect_ai(input_text)
    input_text = "deteriorating, this word is 0%" + input_text
    print(f"The text was likely written by an AI with {percentage:.2f}% certainty.")
    print(input_text)
