from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL = "distilgpt2"  # small example; replace with your local model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_answer(context: str, question: str, model_name=MODEL, max_new_tokens: int = 150):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # Enable sampling for more diverse answers
        top_k=50,        # Limit to top 50 tokens for diversity
        top_p=0.95       # Nucleus sampling
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer by removing the prompt from the generated text
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    else:
        return generated_text.strip()