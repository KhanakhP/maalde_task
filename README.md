# AI-Based Dress Demand Prediction System

## Full Name
Khanakh Prajapati

## Mobile Number
9313212890

---

# Project Overview

This project is an AI/ML-based Dress Demand Prediction System that predicts the expected quantity sold for a new dress design using:

- Dress Images
- Historical Sales Data
- Product Pricing Information

The system uses computer vision, embedding extraction, dimensionality reduction, similarity retrieval, and machine learning to estimate demand.

---

# Features

## Dress Validation
Before prediction, the system validates whether the uploaded image is actually a dress-related image.

Examples:
- Dress → Accepted
- Saree → Accepted
- T-shirt → Rejected
- Sunglasses → Rejected

---

## OCR-Based Image Renaming
The system extracts dress codes from images using OCR and renames images automatically.

---

## Embedding Generation
The project uses FashionCLIP to extract semantic image embeddings representing:
- Color
- Pattern
- Style
- Design Features

---

## PCA Dimensionality Reduction
Since embeddings are high-dimensional and dataset size is small, PCA is used to:
- Reduce noise
- Improve generalization
- Reduce overfitting

Two PCA configurations were tested:
- PCA-32
- PCA-64

---

## Demand Prediction
The final prediction system uses:
- XGBoost Regression
- Similarity-Based Retrieval

Final prediction is generated using a hybrid approach:
- 70% XGBoost prediction
- 30% Similarity retrieval prediction

---

## Similarity Retrieval
The system retrieves visually similar dresses from historical data using cosine similarity on PCA embeddings.

This improves:
- Explainability
- Stability
- Small dataset performance

---

## Streamlit UI
A simple UI is built using Streamlit where users can:
1. Upload dress image
2. Enter product price
3. Get predicted quantity sold
4. View similar historical dresses

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Streamlit | UI |
| FashionCLIP | Image embeddings |
| XGBoost | Demand prediction |
| PCA | Dimensionality reduction |
| Scikit-learn | ML utilities |
| OpenCV | Image preprocessing |
| Tesseract OCR | Code extraction from images |
| NumPy/Pandas | Data handling |

---

# Challenges Faced

```text

Very small dataset size
Noisy demand labels
Weak correlation between visual features and sales
OCR inconsistencies in image codes
Overfitting risk due to high-dimensional embeddings