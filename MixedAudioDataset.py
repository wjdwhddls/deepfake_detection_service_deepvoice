import librosa
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset

from utils import *

# Custom Dataset for mixing data
class MixedAudioDataset(Dataset):
    def __init__(self, real_file_list, fake_file_list, base_path, target_sr, max_sample = 50000):
        # Real and Fake file list
        self.real_file_list = real_file_list
        self.fake_file_list = fake_file_list

        # Base directory of data
        self.base_path = base_path

        # Not Actually Used
        self.max_sample = max_sample

        # Target sampling rate and Sequence Length
        self.target_sr = 16000
        self.seq_len = target_sr * 5

        self.labels = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 1]
        ])

        return

    def __len__(self):
        return self.max_sample

    # Sample random component from input buffer
    def sample_data(self, file_list):
        return file_list[np.random.randint(0, file_list.shape[0])]

    # Apply zero padding into fixed length
    def zero_pad(self, data):
        if data.shape[0] > self.seq_len:
            start_idx = np.random.randint(0, data.shape[0] - self.seq_len)

            return data[start_idx : start_idx + self.seq_len]

        basis = np.zeros(self.seq_len)

        start_idx = np.random.randint(0, self.seq_len - data.shape[0])
        basis[start_idx : start_idx + data.shape[0]] = data

        return basis

    # Apply tile padding into fixed length
    def tile_pad(self, data):
        data = np.concatenate([data for _ in range(self.seq_len // data.shape[0] + 1)])

        start_idx = np.random.randint(0, data.shape[0] - self.seq_len)
        data = data[start_idx : start_idx + self.seq_len]

        return data

    def __getitem__(self, dummy):
        # Data Category
        status = np.random.randint(0, 6)

        # If sampled category is 0, just mix two random noise
        if status == 0:  # [0, 0], Fake: 0, Real: 0
            noise = get_noise(self.seq_len)
            noise_scale = np.random.random()
            noise_occurrence = round(np.random.random())

            data = noise * noise_scale * noise_occurrence
            sr = 32000

        # If sampled category is 1, mix fake voice and random noise
        elif status == 1:  # [1, 0], Fake: 1, Real: 0
            data, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.fake_file_list)}.ogg', sr = self.target_sr)

            # Random Padding Method
            if np.random.randint(0, 2) == 1: # Tile Padding
                data = self.tile_pad(data)
            else: # Zero Padding
                data = self.zero_pad(data)

        # If sampled category is 2, mix real voice and random noise
        elif status == 2:  # [0, 1], Fake: 0, Real: 1
            data, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.real_file_list)}.ogg', sr = self.target_sr)

            # Random Padding Method
            if np.random.randint(0, 2) == 1: # Tile Padding
                data = self.tile_pad(data)
            else: # Zero Padding
                data = self.zero_pad(data)

        # If sampled category is 3, mix fake, real voices and random noise
        elif status == 3:  # [1, 1], Fake: 1, Real: 1
            data_1, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.fake_file_list)}.ogg', sr = self.target_sr)
            data_2, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.real_file_list)}.ogg', sr = self.target_sr)

            data = join_audio(data_1, data_2)

        # If sampled category is 4, mix two fake voices and random noise
        elif status == 4:  # [1, 0], Fake: 2, Real: 0
            data_1, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.fake_file_list)}.ogg', sr = self.target_sr)
            data_2, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.fake_file_list)}.ogg', sr = self.target_sr)

            data = join_audio(data_1, data_2)

        # If sampled category is 5, mix two real voices and random noise
        elif status == 5:  # [0, 1], Fake: 0, Real: 2
            data_1, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.real_file_list)}.ogg', sr = self.target_sr)
            data_2, sr = librosa.load(self.base_path + f'train/{self.sample_data(self.real_file_list)}.ogg', sr = self.target_sr)

            data = join_audio(data_1, data_2)

        # Label to predict
        include_fake, include_real = self.labels[status]

        # Scale the data to its absolute maximum value be 1
        if data.sum() != 0:
            data = data / np.abs(data).max()

        # Scale the overall value of data
        if round(np.random.random()) == 0:
            data = data * np.random.random()

        # Apply Random Noise
        data = apply_noise(data)

        # Scale the data to its absolute maximum value be 1
        if data.sum() != 0:
            data = data / np.abs(data).max()

        # Final Scale Parameter which make the value smaller, but not to be so small
        scale_param = np.abs(np.random.normal(0, 0.25))

        # Scale the final data
        data = data * (1 - scale_param)

        return torch.tensor(data, dtype = torch.float32),  torch.tensor([include_fake, ], dtype = torch.float32), torch.tensor([include_real, ], dtype = torch.float32), status

# Load every audio filename in the data directory
def load_path_list(base_path, train = True):
    if not train:
        test_df = pd.read_csv(base_path + f'sample_submission.csv')

        return test_df['id'].to_numpy()

    train_df = pd.read_csv(base_path + f'train.csv')

    real_file_list = train_df[train_df['label'] == 'real']['id'].to_numpy()
    fake_file_list = train_df[train_df['label'] == 'fake']['id'].to_numpy()

    return real_file_list, fake_file_list

if __name__ == '__main__':

    # Base directory of data
    base_path = f'./data/open/'

    # Output directory
    out_path = f'./data/open/train_aug/'

    # Load every file list
    real_file_list, fake_file_list = load_path_list(base_path, train=True)
    test_file_list = load_path_list(base_path, train=False)

    # Set target sampling rate
    target_sr = 16000

    # Define custom dataset instance
    dummy_dataset = MixedAudioDataset(real_file_list, fake_file_list, base_path, target_sr)

    # For naming the augmented data
    saving_status = [int() for _ in range(6)]

    # Data generation(augmentation)
    for data, include_fake, include_real, status in tqdm(dummy_dataset, desc = 'Augmenting...'):
        sf.write(out_path + f'{status}/{saving_status[status]}.ogg', data, target_sr)
        saving_status[status] += 1