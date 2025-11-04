import json

# Path to your dataset
input_path = "Dataset/own.json"
output_path = "Dataset/combined.json"

# Load the cleaned dataset
with open(input_path, "r") as f:
    data = json.load(f)

combined_data = []

# Loop through each prompt–completion pair
for item in data:
    prompt = item["prompt"].strip()
    completion = item["completion"].strip()
    
    # Combine them into a single text format for GPT training
    combined_text = f"You: {prompt}\nBot: {completion}"
    
    combined_data.append({"text": combined_text})

# Save the concatenated version
with open(output_path, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"✅ Combined dataset saved to {output_path}")
print(f"Total records: {len(combined_data)}")
