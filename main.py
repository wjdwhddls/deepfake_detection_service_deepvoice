import pickle
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch.nn as nn

from utils import *
from MainDataset import *
from model import SiamMetricNetworks

# Load configuration flie
cfg = load_config(f'./config.yaml')

# Set Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basis of the data directory
base_path = './data/open/'

# Load data path list
(anchor_path_list, anchor_label_list), fake_file_list, real_file_list = load_path_list(base_path)

# If there are previous split data file, load it.
if 'train_valid_test.pickle' in os.listdir('./history/'):
    path_dict = pickle.load(open(f'./history/train_valid_test.pickle', 'rb'))

    train_anchor_path_list = path_dict['anchor']['train']['path']
    valid_anchor_path_list = path_dict['anchor']['valid']['path']
    test_anchor_path_list = path_dict['anchor']['test']['path']

    train_anchor_label_list = path_dict['anchor']['train']['label']
    valid_anchor_label_list = path_dict['anchor']['valid']['label']
    test_anchor_label_list = path_dict['anchor']['test']['label']

    train_fake_file_list = path_dict['fake']['train']
    valid_fake_file_list = path_dict['fake']['valid']
    test_fake_file_list = path_dict['fake']['test']

    train_real_file_list = path_dict['real']['train']
    valid_real_file_list = path_dict['real']['valid']
    test_real_file_list = path_dict['real']['test']

    print(f'Loaded Previous Data Setting!')

# Split every data with certain proportion
# Train : Valid : Test = 6 : 2 : 2
else:
    # Train Valid Test Split, Stratifying with Anchor Label
    train_anchor_path_list, valid_anchor_path_list, train_anchor_label_list, valid_anchor_label_list = train_test_split(anchor_path_list, anchor_label_list, test_size = 0.4, shuffle = True, stratify = anchor_label_list)
    valid_anchor_path_list, test_anchor_path_list, valid_anchor_label_list, test_anchor_label_list = train_test_split(valid_anchor_path_list, valid_anchor_label_list, test_size = 0.5, shuffle = True, stratify = valid_anchor_label_list)

    train_fake_file_list, valid_fake_file_list = train_test_split(fake_file_list, test_size = 0.4, shuffle = True)
    valid_fake_file_list, test_fake_file_list = train_test_split(valid_fake_file_list, test_size = 0.5, shuffle = True)

    train_real_file_list, valid_real_file_list = train_test_split(real_file_list, test_size = 0.4, shuffle = True)
    valid_real_file_list, test_real_file_list = train_test_split(valid_real_file_list, test_size = 0.5, shuffle = True)

    # Dictionary for saving
    path_dict = {
        'anchor': {
            'train': {
                'path': train_anchor_path_list,
                'label': train_anchor_label_list
            },
            'valid': {
                'path': valid_anchor_path_list,
                'label': valid_anchor_label_list
            },
            'test': {
                'path': test_anchor_path_list,
                'label': test_anchor_label_list
            }
        },
        'fake': {
            'train': train_fake_file_list,
            'valid': valid_fake_file_list,
            'test': test_fake_file_list
        },
        'real': {
            'train': train_real_file_list,
            'valid': valid_real_file_list,
            'test': test_real_file_list
        }
    }

    # Dump the data split setting
    pickle.dump(path_dict, open(f'./history/train_valid_test.pickle', 'wb'))

    print(f'Data Split with New Setting!')

# Set anchor dataset. It returns anchor embeddings, fake labels, real labels.
train_anchor_dataset = AnchorDataset(train_anchor_path_list, train_anchor_label_list, base_path)
valid_anchor_dataset = AnchorDataset(valid_anchor_path_list, valid_anchor_label_list, base_path)

# Set fake reference dataset. It returns just fake voice embeddings.
train_fake_dataset = ReferenceDataset(train_fake_file_list, base_path, shuffle = True)
valid_fake_dataset = ReferenceDataset(valid_fake_file_list, base_path, shuffle = True)

# Set rake reference dataset. It returns just rake voice embeddings.
train_real_dataset = ReferenceDataset(train_real_file_list, base_path, shuffle = True)
valid_real_dataset = ReferenceDataset(valid_real_file_list, base_path, shuffle = True)

# Set the Dataloader for mixed anchor dataset with 64 batch-size.
train_anchor_loader = DataLoader(train_anchor_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True, drop_last = True)
valid_anchor_loader = DataLoader(valid_anchor_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True, drop_last = True)

# Set the Dataloader for fake reference dataset. Not actually used because of the data length.
train_fake_loader = DataLoader(train_fake_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True)
valid_fake_loader = DataLoader(valid_fake_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True)

# Set the Dataloader for real reference dataset. Not actually used because of the data length.
train_real_loader = DataLoader(train_real_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True)
valid_real_loader = DataLoader(valid_real_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True)

# Define model, optimizer, and criterion
model = SiamMetricNetworks(cfg['INPUT_DIM'], cfg['HIDDEN_DIM'], cfg['OUTPUT_DIM']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = cfg['LEARNING_RATE'])
criterion = nn.BCELoss().to(device)

# Variable that contains lowest validation loss
lowest_valid_loss = np.inf

# Buffer that contains every train and validation loss
train_loss, valid_loss = list(), list()

# Epoch is set 200 in configuration file
for epoch in range(cfg['EPOCHS']):

    # Turn model into train mode
    model.train()

    # Variable that would accumulate every train loss
    running_loss = float()

    # Load Every Anchor Embedding and Labels
    for anchor, include_fake, include_real in tqdm(train_anchor_loader, desc = f'Epoch {epoch} Training...'):

        # Set the data location(cpu or gpu)
        anchor = anchor.to(device)
        include_fake, include_real = include_fake.to(device), include_real.to(device)

        # Load every reference embedding.
        ref_fake, ref_real = list(), list()
        for i in range(anchor.shape[0]):
            ref_fake.append(train_fake_dataset[i].unsqueeze(0))
            ref_real.append(train_real_dataset[i].unsqueeze(0))
        ref_fake, ref_real = torch.cat(ref_fake, dim=0).to(device), torch.cat(ref_real, dim=0).to(device)

        # Make prediction
        pred_fake, pred_real = model(anchor, ref_fake, ref_real)

        # Calculate Loss
        loss_fake = criterion(pred_fake, include_fake)
        loss_real = criterion(pred_real, include_real)

        # Merge the two loss
        loss = loss_fake + loss_real

        # Update Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the train loss
        running_loss += loss.item()

    # Append average train loss into buffer
    train_loss.append(running_loss / len(train_anchor_loader))

    # Turn model into train mode
    model.eval()

    # Variable that would accumulate every validation loss
    running_loss = float()

    with torch.no_grad():

        # Load Every Anchor Embedding and Labels
        for anchor, include_fake, include_real in tqdm(valid_anchor_loader, desc = f'Epoch {epoch} Validating...'):

            # Set the data location(cpu or gpu)
            anchor = anchor.to(device)
            include_fake, include_real = include_fake.to(device), include_real.to(device)

            # Load every reference embedding.
            ref_fake, ref_real = list(), list()
            for i in range(anchor.shape[0]):
                ref_fake.append(valid_fake_dataset[i].unsqueeze(0))
                ref_real.append(valid_real_dataset[i].unsqueeze(0))
            ref_fake, ref_real = torch.cat(ref_fake, dim=0).to(device), torch.cat(ref_real, dim=0).to(device)

            # Make prediction
            pred_fake, pred_real = model(anchor, ref_fake, ref_real)

            # Calculate Loss
            loss_fake = criterion(pred_fake, include_fake)
            loss_real = criterion(pred_real, include_real)

            # Merge the two loss
            loss = loss_fake + loss_real

            # Accumulate the validation loss
            running_loss += loss.item()

    # Append average validation loss into buffer
    valid_loss.append(running_loss / len(valid_anchor_loader))

    # Logging
    print(f'Epoch {epoch} | Train Loss: {train_loss[-1]:.3f}, Valid Loss: {valid_loss[-1]:.3f}')

    # If the lowest validation loss was renewed, save the parameters
    if lowest_valid_loss > valid_loss[-1]:
        # Logging
        print(f'!!! Lowest Validation Loss Renewed! {lowest_valid_loss} -> {valid_loss[-1]}')

        # Set the lowest validation loss to current validation loss
        lowest_valid_loss = valid_loss[-1]

        # Save each state dictionary of model and optimizer
        torch.save(model.state_dict(), f'./history/model/Epoch_{epoch}_best_model.pth')
        torch.save(optimizer.state_dict(), f'./history/optimizer/Epoch_{epoch}_optimizer.pth')

# Save every loss buffer into history directory
pickle.dump({
    'train_loss': train_loss,
    'valid_loss': valid_loss
}, open(f'./history/history.pickle', 'wb'))