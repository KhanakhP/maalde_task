import torch
from PIL import Image

from transformers import CLIPProcessor
from transformers import CLIPModel

# LOAD MODEL
print("Loading CLIP model...")
print("SCRIPT STARTED")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("CLIP loaded successfully")

# VALID / INVALID LABELS
VALID_LABELS = [
    "a traditional dress", "a women's dress", "ethnic wear", "a saree",
    "a kurti", "a lehenga", "women's clothing"]

INVALID_LABELS = [
    "a t-shirt", "a shirt", "sunglasses", "a shoe", "a bag","electronics", 
    "a mobile phone", "a watch", "furniture", "a laptop"]

ALL_LABELS = VALID_LABELS + INVALID_LABELS

# VALIDATOR FUNCTION
def validate_dress(image_path):
    
    # LOAD IMAGE
    image = Image.open(image_path).convert("RGB")

    # PREPARE INPUTS
    inputs = processor(text=ALL_LABELS,images=image,return_tensors="pt",padding=True)

    # INFERENCE
    with torch.no_grad():

        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image

        probs = logits_per_image.softmax(dim=1)
        
    # GET BEST MATCH
    best_idx = probs.argmax().item()

    best_label = ALL_LABELS[best_idx]

    confidence = probs[0][best_idx].item()

    # VALIDATION LOGIC
    is_valid = (
        best_label in VALID_LABELS
        and confidence > 0.30
    )

    # RESULT
    return {
        "is_valid": is_valid,
        "label": best_label,
        "confidence": confidence
    }

# TEST
if __name__ == "__main__":
    print("ENTERING MAIN BLOCK")
    image_path = input("Enter image path: ")

    result = validate_dress(image_path)

    print("\n===== RESULT =====")

    print(f"Prediction : {result['label']}")
    print(f"Confidence : {result['confidence']:.4f}")

    if result["is_valid"]:

        print("VALID DRESS IMAGE")

    else:

        print("INVALID IMAGE")