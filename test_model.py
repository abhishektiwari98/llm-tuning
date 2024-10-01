from transformers import pipeline

# Load the prompt-tuned model and tokenizer
model_path = "./prompt-tuned-model"
nlp = pipeline("text-generation", model=model_path, tokenizer=model_path)

# Test the model with a specific prompt
input_text = "Sentiment of this review is: This movie was not good! ->"
print(nlp(input_text))
