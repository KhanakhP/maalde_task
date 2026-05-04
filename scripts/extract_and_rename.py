# OCR utility for renaming raw product images by detected product code.
import os
import re
import shutil

import cv2
import easyocr
import pytesseract


INPUT_FOLDER = "images"
OUTPUT_FOLDER = "renamed_images"
FAILED_FOLDER = "failed_images"


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh


reader = easyocr.Reader(["en"], gpu=False)


def extract_code_from_image(image_path):
    processed_img = preprocess_image(image_path)

    try:
        text = pytesseract.image_to_string(processed_img)
        matches = re.findall(r"\b\d{8}\b", text)
        if matches:
            return matches[-1]
    except Exception as exc:
        print(f"[Tesseract Error] {image_path}: {exc}")

    try:
        results = reader.readtext(image_path, detail=0)
        text = " ".join(results)
        matches = re.findall(r"\b\d{8}\b", text)
        if matches:
            return matches[-1]
    except Exception as exc:
        print(f"[EasyOCR Error] {image_path}: {exc}")

    return None


def process_images():
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_path = os.path.join(INPUT_FOLDER, filename)

        try:
            code = extract_code_from_image(input_path)

            if code:
                output_path = os.path.join(OUTPUT_FOLDER, f"{code}.jpg")

                counter = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(OUTPUT_FOLDER, f"{code}_{counter}.jpg")
                    counter += 1

                shutil.copy(input_path, output_path)
                print(f"[SUCCESS] {filename} -> {os.path.basename(output_path)}")
            else:
                shutil.copy(input_path, os.path.join(FAILED_FOLDER, filename))
                print(f"[FAILED OCR] {filename}")
        except Exception as exc:
            shutil.copy(input_path, os.path.join(FAILED_FOLDER, filename))
            print(f"[ERROR] {filename} | {exc}")


if __name__ == "__main__":
    process_images()
