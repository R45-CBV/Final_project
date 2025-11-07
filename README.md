# 🧠 Proprietary Company Support Chatbot

This project is a **domain-specific support chatbot** trained on **internal company knowledge-base data**.  
It provides accurate responses to organization-related queries while ensuring full **data confidentiality** and **no external API usage**.

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| **Custom Fine-Tuned Model** | Model is trained on internal Q&A datasets to ensure accurate support responses. |
| **Offline & Secure** | Runs locally — no external LLM calls. Data stays within the company. |
| **Contextual Responses** | Maintains short conversation history for more natural dialogue. |
| **Safety Filter** | Rejects queries outside the supported domain to avoid incorrect / hallucinated answers. |
| **Easy to Extend** | New Q&A records can be added to improve performance over time. |

---

## 🏗️ Project Structure

Project/
│
├── Dataset/
│ ├── Raw/ # Original collected company data
│ ├── Cleaned/ # Cleaned and formatted Q&A JSON pairs
│ └── Tokenized/ # Tokenized dataset used for training
│
├── Model/
│ └── FineTuned_Model/ # Final trained model + tokenizer
│
├── Training/
│ └── train_model.py # Model fine-tuning script
│
├── Chat/
│ └── run_chat.py # Interactive chatbot testing script---

**⚙️ Setup Instructions**

1. Create & Activate Virtual Environment
python3 -m venv env
source env/bin/activate

2. Install Dependencies
pip install torch transformers datasets

3. Run Chatbot Locally
python Chat/run_chat.py

📊 **Model Information**

Property	          Value
Base Model	        distilgpt2
Training Method	    Supervised fine-tuning
Dataset Format	    JSON in prompt → response format
Recommended Epochs  4–8 based on dataset size
Tokenizer Used	    GPT2TokenizerFast│


