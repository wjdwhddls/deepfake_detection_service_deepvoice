import os
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import *
from MainDataset import *
from model import SiamMetricNetworks

if len(os.listdir(f'./history/model/')) == 0:
    print(f'There are no parameters to test. Please run this code after full training!')
    exit()

cfg = load_config(f'./config.yaml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_path = './data/open'

(anchor_path_list, anchor_label_list), fake_file_list, real_file_list = load_path_list(base_path)

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

anchor_dataset = AnchorDataset(test_anchor_path_list, test_anchor_label_list, base_path)

fake_dataset = ReferenceDataset(test_fake_file_list, base_path, shuffle = True)
real_dataset = ReferenceDataset(test_real_file_list, base_path, shuffle = True)

anchor_loader = DataLoader(anchor_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True, drop_last = False)

# fake_loader = DataLoader(fake_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True)
# real_loader = DataLoader(real_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = True)

# Pre-Sampling
fake_loader, real_loader = list(), list()
for idx in range(len(anchor_loader)):
    ref_fake, ref_real = list(), list()
    for i in range(cfg['BATCH_SIZE']):
        ref_fake.append(fake_dataset[i].unsqueeze(0))
        ref_real.append(real_dataset[i].unsqueeze(0))
    ref_fake, ref_real = torch.cat(ref_fake, dim=0), torch.cat(ref_real, dim=0)
    fake_loader.append(ref_fake)
    real_loader.append(ref_real)

ckp_list = sorted([int(val.split(f'_')[1]) for val in os.listdir(f'./history/model/')])
ckp_list = [f'Epoch_{val}_best_model.pth' for val in ckp_list]

model = SiamMetricNetworks(cfg['INPUT_DIM'], cfg['HIDDEN_DIM'], cfg['OUTPUT_DIM']).to(device)
criterion = nn.BCELoss().to(device)

loss_dict = dict()
precision_dict = dict()
recall_dict = dict()
f1_dict = dict()
accuracy_dict = dict()

with torch.no_grad():
    for ckp_name in ckp_list:
        msg = model.load_state_dict(torch.load(f'./history/model/{ckp_name}', map_location = device))
        print(f'{ckp_name} | {msg}')

        total_gt_fake = list()
        total_gt_real = list()

        total_pred_fake = list()
        total_pred_real = list()

        running_loss = float()

        for (anchor, include_fake, include_real), ref_fake, ref_real in tqdm(zip(anchor_loader, fake_loader, real_loader), desc=f'CKP {ckp_name.split(f"_")[1]} Testing...'):
            anchor = anchor.to(device)
            include_fake, include_real = include_fake.to(device), include_real.to(device)

            ref_fake = ref_fake[:anchor.shape[0]].to(device)
            ref_real = ref_real[:anchor.shape[0]].to(device)

            # ref_fake, ref_real = list(), list()
            # for i in range(anchor.shape[0]):
            #     ref_fake.append(fake_dataset[i].unsqueeze(0))
            #     ref_real.append(real_dataset[i].unsqueeze(0))
            # ref_fake, ref_real = torch.cat(ref_fake, dim=0).to(device), torch.cat(ref_real, dim=0).to(device)

            pred_fake, pred_real = model(anchor, ref_fake, ref_real)

            loss_fake = criterion(pred_fake, include_fake)
            loss_real = criterion(pred_real, include_real)

            loss = loss_fake + loss_real

            running_loss += loss.item()

            total_gt_fake.append(include_fake.cpu().detach().numpy())
            total_gt_real.append(include_real.cpu().detach().numpy())

            total_pred_fake.append(pred_fake.cpu().detach().numpy())
            total_pred_real.append(pred_real.cpu().detach().numpy())

        total_gt_fake = np.concatenate(total_gt_fake, axis = 0)
        total_gt_real = np.concatenate(total_gt_real, axis = 0)

        total_pred_fake = (np.concatenate(total_pred_fake, axis = 0) > 0.5).astype(np.float32)
        total_pred_real = (np.concatenate(total_pred_real, axis = 0) > 0.5).astype(np.float32)

        fake_precision = len(np.where((total_pred_fake == 1.0) & (total_gt_fake == 1.0))[0]) / len(np.where(total_pred_fake == 1.0)[0])
        real_precision = len(np.where((total_pred_real == 1.0) & (total_gt_real == 1.0))[0]) / len(np.where(total_pred_real == 1.0)[0])

        fake_recall = len(np.where((total_pred_fake == 1.0) & (total_gt_fake == 1.0))[0]) / len(np.where(total_gt_fake == 1.0)[0])
        real_recall = len(np.where((total_pred_real == 1.0) & (total_gt_real == 1.0))[0]) / len(np.where(total_gt_real == 1.0)[0])

        fake_f1 = 2 * (fake_precision * fake_recall) / (fake_precision + fake_recall)
        real_f1 = 2 * (real_precision * real_recall) / (real_precision + real_recall)

        fake_accuracy = (total_pred_fake == total_gt_fake).mean()
        real_accuracy = (total_pred_real == total_gt_real).mean()

        print(f'Fake | Precision: {fake_precision}, Recall: {fake_recall}, F1: {fake_f1}, Accuracy: {fake_accuracy}')
        print(f'Real | Precision: {real_precision}, Recall: {real_recall}, F1: {real_f1}, Accuracy: {real_accuracy}')

        loss_dict[ckp_name] = running_loss / len(anchor_loader)
        precision_dict[ckp_name] = [fake_precision, real_precision]
        recall_dict[ckp_name] = [fake_recall, real_recall]
        f1_dict[ckp_name] = [fake_f1, real_f1]
        accuracy_dict[ckp_name] = [fake_accuracy, real_accuracy]


pickle.dump(loss_dict, open(f'./history/test_loss.pickle', 'wb'))
pickle.dump(precision_dict, open(f'./history/test_precision.pickle', 'wb'))
pickle.dump(recall_dict, open(f'./history/test_recall.pickle', 'wb'))
pickle.dump(f1_dict, open(f'./history/test_f1.pickle', 'wb'))
pickle.dump(accuracy_dict, open(f'./history/test_accuracy.pickle', 'wb'))