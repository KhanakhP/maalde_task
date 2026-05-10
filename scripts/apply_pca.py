import os 
import numpy as np 
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#paths
embeddings_path = 'data/embeddings/fashion_clip_embeddings.npy'
output_dir = 'data/embeddings'
pca_dir = 'data/pca'

#create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pca_dir, exist_ok=True)

#load embeddings
embeddings = np.load(embeddings_path)
print(f"Original embeddings shape: {embeddings.shape}")

#standardize features
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

#save scaler
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

#pca config
pca_configs = [32, 64]

#apply PCA
for n_components in pca_configs:
    print(f"Applying PCA with {n_components} components")
    #initialize PCA
    pca = PCA(n_components=n_components, random_state=42)
    #fit and transform
    reduced_embeddings = pca.fit_transform(embeddings_scaled)
    #save reduced embeddings
    output_path = os.path.join(output_dir, f'f_clip_pca_{n_components}_embedding.npy')
    np.save(output_path, reduced_embeddings)
    #save PCA model
    pca_model_path = os.path.join(pca_dir, f'pca_{n_components}_model.pkl')
    joblib.dump(pca, pca_model_path)
    
    #summary
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Reduce embeddings shape: {reduced_embeddings.shape}")
    print(f"Explained variance ratio: {explained_variance:.4f}")
    
print("\nPCA application completed.")
