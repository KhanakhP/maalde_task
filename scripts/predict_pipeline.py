import numpy as np 
import joblib
from fashion_clip.fashion_clip import FashionCLIP
from scripts.dress_validator import validate_dress
from scripts.similarity_predictor import similarity_predict

# LOAD MODELS
print("Loading models...")

fclip = FashionCLIP("fashion-clip")
pca = joblib.load("data/pca/pca_32_model.pkl")
scaler = joblib.load("data/embeddings/scaler.pkl")
model = joblib.load("models/xgb_pca32.pkl")
rate_scaler = joblib.load("models/rate_scaler.pkl")

print("Models loaded successfully")

#prediction function
def predict_demand(image_path, rate):
    #validate image
    validation = validate_dress(image_path)
    if not validation["is_valid"]:
        return{"success":False, "message":("Invalid image." "Please upload dress-related image"), "prediction":None}
    
    #Generate embeddings
    embedding = fclip.encode_images([image_path], batch_size=1)
    embedding = embedding.squeeze()
    #Scale
    embedding_scaled = scaler.transform(embedding.reshape(1,-1))
    #PCA transform
    reduced_embedding = pca.transform(embedding_scaled)
    #rate features
    log_rate = np.log1p(rate)
    normalized_rate = rate_scaler.transform([[rate]])[0][0]
    rate_features = np.array([[rate, log_rate, normalized_rate]])
    #final feature vector
    X = np.concatenate([reduced_embedding, rate_features], axis=1)
    #predict
    pred_log = model.predict(X)[0]
    # ======================================
    # SIMILARITY PREDICTION
    # ======================================

    similarity_result = similarity_predict(
        reduced_embedding[0],
        rate,
        top_k=5
    )

    similarity_prediction = (
        similarity_result["prediction"]
    )
    xgb_prediction = np.expm1(pred_log)

    prediction = (
        0.7 * xgb_prediction
        +
        0.3 * similarity_prediction
    )
    prediction = max(0, prediction)
    #result
    return{"success":True, "message":"Prediction successful", "prediction":int(prediction), "similar_items":similarity_result["similar_items"], "label":validation["label"], "confidence":validation["confidence"]}

#test
if __name__ == "__main__":
    image_path = input("Enter image path: ")
    rate = float(input("Enter product rate: "))
    result = predict_demand(
        image_path,
        rate
    )
    print("\n===== RESULT =====")
    print(result)