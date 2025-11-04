import json
from sklearn.model_selection import train_test_split

# Load your combined dataset
with open("Dataset/combined.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Split data (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save to new files
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

print(f"✅ Training samples: {len(train_data)}")
print(f"✅ Testing samples: {len(test_data)}")
