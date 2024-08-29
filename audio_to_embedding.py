import torch

import os
from tqdm.auto import tqdm

import wespeaker as ws

# Directory where we downloaded the pretrained model
model_path = f'./WeSpeaker_ResNet221/'

# Basis of the data directory
base_path = './data/open/'

# Load model and set GPU
model = ws.load_model_local(model_path)
model.set_gpu(0 if torch.cuda.is_available() else -1)

# Embed every augmented data for training(Anchor)
for label in range(6):
    for file_name in tqdm(os.listdir(base_path + f'train_aug/{label}/'), desc = f'Train | Label {label} Embedding...'):
        embedding = model.extract_embedding(base_path + f'train_aug/{label}/{file_name}')
        torch.save(embedding, base_path + f'train_aug_emb/{label}/{file_name.split(".")[0]}.pth')

# Embed every single train audio data(Reference)
for file_name in tqdm(os.listdir(base_path + f'train/'), desc = 'Train | Embedding...'):
    embedding = model.extract_embedding(base_path + f'train/{file_name}')
    torch.save(embedding, base_path + f'train_emb/{file_name.split(".")[0]}.pth')

# Embed every test data for inference session
for file_name in tqdm(os.listdir(base_path + f'test/'), desc = 'Test | Embedding...'):
    embedding = model.extract_embedding(base_path + f'test/{file_name}')
    torch.save(embedding, base_path + f'test_emb/{file_name.split(".")[0]}.pth')