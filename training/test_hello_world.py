import sys
from transformers import pipeline

if len(sys.argv) < 2:
    print("Usage: python test_model.py 'your text here'")
    sys.exit(1)

test_text = sys.argv[1]

# Load model with pipeline
classifier = pipeline("text-classification", model="./my_trained_model", device=0)

# Get prediction
result = classifier(test_text)

print(f"\nText: {test_text}")
print(f"Prediction: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.4f}")
