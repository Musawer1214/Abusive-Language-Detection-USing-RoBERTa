import gradio as gr
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

# Load the model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("./")  # Load model from the current directory
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")    # Use the pre-trained tokenizer

# Define a function to make predictions
def detect_abusive_language(text):
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    
    # Get the prediction (0 for non-abusive, 1 for abusive)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Abusive language detected!" if prediction == 1 else "No abusive language detected."

# Define Gradio interface
interface = gr.Interface(
    fn=detect_abusive_language,
    inputs="text",
    outputs="text",
    title="Abusive Language Detection",
    description="Enter text to check if it contains abusive language."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
