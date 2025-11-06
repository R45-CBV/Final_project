from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
model_path = "Finetuned_old/FineTuned_Model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure model runs efficiently
model.eval()

# Simple chat loop
print("🧠 SOULSYNC Chatbot (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye 👋")
        break

    prompt = f"You: {user_input}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
    **inputs,
    max_length=150,
    pad_token_id=tokenizer.eos_token_id,
    temperature=1.6,        # 🧨 Encourages hallucinations
    top_p=0.98,             # 🧨 Allows very wide possible token choices
    repetition_penalty=1.0, # ❌ No protection against looping/wrong info
    do_sample=True
)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)       
    bot_reply = response.split("Bot:")[-1].strip()
    print(f"Bot: {bot_reply}\n")
