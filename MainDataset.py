import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for loading Mixed Audio Embedding
class AnchorDataset(Dataset): # Mixed Data
    def __init__(self, path_list, label_list, base_path):
        self.base_path = base_path

        self.path_list = path_list
        self.label_list = label_list

        self.mapping = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 1]
        ])

        return

    def __len__(self):
        return self.path_list.shape[0]

    def __getitem__(self, idx):
        embedding = torch.load(self.base_path + self.path_list[idx])
        label = int(self.label_list[idx])

        include_fake, include_real = self.mapping[label]

        return embedding, torch.tensor([include_fake, ], dtype = torch.float32), torch.tensor([include_real, ], dtype = torch.float32)

# Custom Dataset for loading Reference Audio Embedding(Pure Real or Fake Voice)
class ReferenceDataset(Dataset): # Real or Fake Dataset
    def __init__(self, path_list, base_path, shuffle = False):
        self.base_path = base_path

        self.path_list = path_list
        self.shuffle = shuffle

        self.seed = None

        return

    def __len__(self):
        return self.path_list.shape[0]

    def set_seed(self, seed):
        self.seed = seed

    def __getitem__(self, idx):
        if self.shuffle:
            if type(self.seed) != type(None):
                np.random.seed(self.seed[idx])

            return torch.load(self.base_path + f'train_emb/{self.path_list[np.random.randint(0, len(self.path_list))]}.pth')

        embedding = torch.load(self.base_path + f'train_emb/{self.path_list[idx]}.pth')

        return embedding

# Custom Dataset for loading Test Data
class InferenceDataset(Dataset): # Test Data
    def __init__(self, path_list, base_path):
        self.base_path = base_path

        self.path_list = path_list

        return

    def __len__(self):
        return self.path_list.shape[0]

    def __getitem__(self, idx):
        embedding = torch.load(self.base_path + f'test_emb/{self.path_list[idx]}.pth')

        return embedding

# Load every embedding filename in the data directory
def load_path_list(base_path, test = False):

    if test:
        test_df = pd.read_csv(base_path + 'sample_submission.csv')

        return test_df['id'].to_numpy()

    anchor_base_path = base_path + 'train_aug_emb/'

    class_list = os.listdir(anchor_base_path)

    path_list = list()
    label_list = list()

    for class_id in class_list:
        file_list = np.array([f'train_aug_emb/{class_id}/{file_name}' for file_name in os.listdir(base_path + f'train_aug_emb/{class_id}/')])
        path_list.append(file_list)
        label_list.append(np.zeros(*file_list.shape) + int(class_id))

    anchor_path_list = np.concatenate(path_list, axis = 0)
    anchor_label_list = np.concatenate(label_list, axis = 0)

    train_df = pd.read_csv(base_path + 'train.csv')
    fake_file_list = train_df[train_df['label'] == 'fake']['id'].to_numpy()
    real_file_list = train_df[train_df['label'] == 'real']['id'].to_numpy()

    return (anchor_path_list, anchor_label_list), fake_file_list, real_file_list


if __name__ == '__main__':
    base_path = f'./data/open/'

    (path_list, label_list), fake_file_list, real_file_list = load_path_list(base_path)

    dummy_dataset = AnchorDataset(path_list, label_list, base_path)

    for i in range(10):
        embedding, include_fake, include_real = dummy_dataset[i]
        print([embedding.shape, include_fake, include_real])

    print([fake_file_list[:5], real_file_list[:5]])