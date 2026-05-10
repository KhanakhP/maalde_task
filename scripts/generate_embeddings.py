import os
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from fashion_clip.fashion_clip import FashionCLIP

#config
dataset_path = 'data/Final_dataset.csv'
embedding_dir = 'data/embeddings'
embedding_output = os.path.join(embedding_dir, 'fashion_clip_embeddings.npy')
code_output = os.path.join(embedding_dir, 'embedding_codes.npy')

#create output directory
os.makedirs(embedding_dir, exist_ok = True)

#load dataset
df = pd.read_csv(dataset_path)
print(f"Load dataset with {len(df)} rows")

#load fashion clip model
print("\nLoading FashionCLIP model")
fclip = FashionCLIP("fashion-clip")
print("Model loaded successfully")

#generate embeddings
all_embeddings = []
all_codes = []
print("\nGenerating embeddings")
for index, row in tqdm(df.iterrows(), total=len(df)):
    code = row['code']
    image_path = row['image_path']
    #check image exists
    if not os.path.exists(image_path):
        print(f"image not found: {image_path}")
        continue
    try:
        #generate embedding
        embedding = fclip.encode_images([image_path], batch_size=1)
        #embeddiing shape: (1, emb_dim)
        embedding = embedding.squeeze()
        all_embeddings.append(embedding)
        all_codes.append(code)
    except Exception as e:
        print(f"error processing {image_path}: {e}")
        
#convert to numpy
all_embeddings = np.array(all_embeddings)
all_codes = np.array(all_codes)

#save
np.save(embedding_output, all_embeddings)
np.save(code_output, all_codes)

#summary
print("\n##### EMBEDDING SUMMARY #####")
print(f"Total embeddings generated: {len(all_embeddings)}")
print(f"Embedding shape: {all_embeddings.shape}")
print(f"\nSaved to {embedding_output} and {code_output}")
print(all_embeddings[0][:10])
print("Embedding generation completed successfully")

