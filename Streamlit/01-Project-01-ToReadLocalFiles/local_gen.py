from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL = "distilgpt2" # small example; replace with your local model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_answer(context: str, question: str, model_name=MODEL, max_new_tokens: int = 150):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()