import numpy as np 
import pandas as pd 
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

#config
dataset_path = 'data\Final_dataset.csv'
PCA_32_path = ('data\embeddings/f_clip_pca_32_embedding.npy')
PCA_64_path = ('data\embeddings/f_clip_pca_64_embedding.npy')

#load_dataset
df = pd.read_csv(dataset_path)
print(f"Dataset shape: {df.shape}")

#feature engineering
# Basic rate feature
df["avg_rate"] = df["rate"].astype(float)
# Log feature
df["log_rate"] = np.log1p(df["rate"])
# Normalized feature
rate_scaler = StandardScaler()
df["normalized_rate"] = rate_scaler.fit_transform(df[["rate"]])

joblib.dump(rate_scaler, "models/rate_scaler.pkl")

#target
y = np.log1p(df["qty"].values)

#load embeddings
embeddings_config = {
    'pca_32': np.load(PCA_32_path)
}

#cross validation
kf = KFold(n_splits = 5, shuffle=True, random_state=42)

#train loop
for config_name, embeddings in embeddings_config.items():
    print(f"\n======== {config_name} ========")
    #merge features
    rate_features = df[["rate", "log_rate", "normalized_rate"]].values
    
    X_array = np.concatenate([embeddings, rate_features], axis=1)
    # FEATURE NAMES
    embedding_feature_names = [f"embed_{i}" for i in range(embeddings.shape[1])]

    rate_feature_names = ["rate","log_rate","normalized_rate"]

    feature_names = (embedding_feature_names + rate_feature_names)
    # DATAFRAME
    X = pd.DataFrame(X_array,columns=feature_names)
    
    print(f"Final feature shape: {X.shape}")
    
    #metric storage
    val_mae_scores = []
    val_r2_scores = []
    train_mae_scores = []
    train_r2_scores = []
    
    #kfold training
    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        #model
        model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42,
                             monotone_constraints={"rate":-1, "log_rate":-1, "normalized_rate":-1})
        #train
        model.fit(X_train, y_train)
        #predict
        preds_logs = model.predict(X_test)
        preds = np.expm1(preds_logs)
        y_true = np.expm1(y_test)
        
        #train predictions
        train_preds_log = model.predict(X_train)
        train_preds = np.expm1(train_preds_log)
        y_train_true = np.expm1(y_train)
        #validation predictions
        val_preds_log = model.predict(X_test)
        val_preds = np.expm1(val_preds_log)
        y_val_true = np.expm1(y_test)
        
        #train metrics
        train_mae = mean_absolute_error(y_train_true, train_preds)
        train_r2 = r2_score(y_train_true, train_preds)
        #validation metrics
        val_mae = mean_absolute_error(y_val_true, val_preds)
        val_r2 = r2_score(y_val_true, val_preds)
        
        val_mae_scores.append(val_mae)
        val_r2_scores.append(val_r2)
        train_mae_scores.append(train_mae)
        train_r2_scores.append(train_r2)
        
        print("\nTRAIN METRICS")
        print(f"Train MAE : {train_mae:.4f}")
        print(f"Train R2  : {train_r2:.4f}")

        print("\nVALIDATION METRICS")

        print(f"Val MAE : {val_mae:.4f}")
        print(f"Val R2  : {val_r2:.4f}")
        fold += 1
        
    #final results
    print(f"\nTRAIN Average MAE for {config_name}: {np.mean(train_mae_scores):.4f}")
    print(f"TRAIN Average R2 for {config_name}: {np.mean(train_r2_scores):.4f}")
    print(f"\nVAL Average MAE for {config_name}: {np.mean(val_mae_scores):.4f}")
    print(f"VAL Average R2 for {config_name}: {np.mean(val_r2_scores):.4f}")
    
print("\n Model training completed.")
joblib.dump(
    model,
    "models/xgb_pca32.pkl"
)

print("\nModel saved successfully")

