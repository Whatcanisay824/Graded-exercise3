import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_tool import get_context

MODEL_PATH = os.environ.get("MODEL_PATH")
if MODEL_PATH is None:
    if os.path.exists("qwen3-0.6b"):
        MODEL_PATH = "qwen3-0.6b"
    elif os.path.exists("models/Qwen3-0.6B"):
        MODEL_PATH = "models/Qwen3-0.6B"
    else:
        from modelscope import snapshot_download
        MODEL_PATH = snapshot_download("Qwen/Qwen3-0.6B")

print("loading tokenizer from", MODEL_PATH, flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("loading model", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
print("model ready", flush=True)

def answer_question(question):
    context = get_context(question, top_k=5)
    system_prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context does not contain the answer, say "I don't know" or "The knowledge base does not contain that information." Do not make up information.

Context:
{context}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    print("Question:", test_question, flush=True)
    print("Answer:", answer_question(test_question), flush=True)