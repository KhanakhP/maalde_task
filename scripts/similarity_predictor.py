import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATA
# ==========================================

print("Loading similarity database...")

# Reduced PCA embeddings
train_embeddings = np.load(
    "data/embeddings/f_clip_pca_32_embedding.npy"
)

# Dataset
df = pd.read_csv(
    "data/final_dataset.csv"
)

print("Similarity database loaded")

# ==========================================
# SIMILARITY FUNCTION
# ==========================================

def similarity_predict(
    query_embedding,
    rate,
    top_k=5
):

    # ======================================
    # COSINE SIMILARITY
    # ======================================

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        train_embeddings
    )[0]

    # ======================================
    # TOP K MOST SIMILAR
    # ======================================

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    # ======================================
    # RETRIEVE RESULTS
    # ======================================

    similar_items = []

    weighted_sum = 0
    similarity_sum = 0

    for idx in top_indices:

        similarity_score = similarities[idx]

        row = df.iloc[idx]

        qty = row["qty"]

        retrieved_rate = row["rate"]
        
        code = row["code"]

        image_path = row["image_path"]

        # ======================================
        # PRICE-AWARE SIMILARITY
        # ======================================

        price_diff = abs(rate - retrieved_rate)

        price_similarity = 1 / (1 + (price_diff / 1000))

        adjusted_similarity = (
            similarity_score * price_similarity
        )
        
        # ----------------------------------
        # WEIGHTED DEMAND
        # ----------------------------------

        weighted_sum += (
            adjusted_similarity * qty
        )

        similarity_sum += adjusted_similarity

        # ----------------------------------
        # STORE RESULT
        # ----------------------------------

        similar_items.append({

            "code": code,

            "qty": qty,

            "similarity": float(
                adjusted_similarity
            ),

            "image_path": image_path
        })

    # ======================================
    # FINAL PREDICTION
    # ======================================

    if similarity_sum == 0:

        prediction = 0

    else:

        prediction = (
            weighted_sum / similarity_sum
        )

    # ======================================
    # RETURN
    # ======================================

    return {

        "prediction": float(prediction),

        "similar_items": similar_items
    }