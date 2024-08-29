import yaml
import numpy as np

# Function to load 'config.yaml'
def load_config(file_path):
    with open(file_path) as file:
        config = yaml.safe_load(file)

    return config

# Mix two audio with overlapping
def join_audio_overlap(src_1, src_2, seq_len = 16000 * 5):
    if src_1.shape[0] > src_2.shape[0]:
        basis = np.zeros_like(src_1)
        start_idx = np.random.randint(0, src_1.shape[0] - src_2.shape[0])
        basis[start_idx: start_idx + src_2.shape[0]] = src_2

        return src_1 + basis

    if src_1.shape[0] == src_2.shape[0]:
        return src_1 + src_2

    basis = np.zeros_like(src_2)
    start_idx = np.random.randint(0, src_2.shape[0] - src_1.shape[0])
    basis[start_idx: start_idx + src_1.shape[0]] = src_1

    data = basis + src_2

    data = np.concatenate([data for _ in range(seq_len // data.shape[0] + 1)], axis = 0)
    start_idx = np.random.randint(0, data.shape[0] - seq_len)
    data = data[start_idx : start_idx + seq_len]

    return data

# Simply join two audio into fixed length
def join_audio(src_1, src_2, seq_len = 16000 * 5):
    basis_1 = np.zeros(seq_len)
    basis_2 = np.zeros(seq_len)

    src_1 = src_1[:min(seq_len, src_1.shape[0])]
    src_2 = src_2[:min(seq_len, src_2.shape[0])]

    start_idx_1 = np.random.randint(0, seq_len - src_1.shape[0] + 1)
    start_idx_2 = np.random.randint(0, seq_len - src_2.shape[0] + 1)

    basis_1[start_idx_1 : start_idx_1 + src_1.shape[0]] = src_1
    basis_2[start_idx_2 : start_idx_2 + src_2.shape[0]] = src_2

    return basis_1 + basis_2

# Apply periodicity to generated noise
def periodic_noise(seq_len, x_shift, x_scale):
    noise = (np.sin(np.linspace(0, 2 * np.pi, seq_len) * x_scale + x_shift) + 1) / 2

    return noise

# Generate random noise
def get_noise(size):
    noise = np.random.normal(0, 1, size=size)
    noise = noise / np.abs(noise).max()

    x_shift = np.abs(np.random.normal())
    x_scale = np.random.random() * 20

    periodic_occurrence = bool(round(np.random.random()))

    if periodic_occurrence:
        noise_p = periodic_noise(size, x_shift, x_scale)
        noise = noise * noise_p

    return noise

# Apply random noise to input sequence
def apply_noise(src):
    noise = get_noise(src.shape[0])
    noise_scale = np.random.random()
    noise_occurrence = round(np.random.random())

    return src + noise * noise_scale * noise_occurrence