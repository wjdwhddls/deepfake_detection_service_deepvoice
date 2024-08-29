import pickle
from tqdm.auto import tqdm

from utils import *
from MainDataset import *
from model import SiamMetricNetworks

# To reproduce our final submission, keep this parameter 'True'
reproduce = True

# Set Threshold or not
apply_threshold = False
thres = 0.5

# References to load for each test sample
n_heads = 6

# Prevent overwriting
inference_id = len(os.listdir(f'./submission/'))

# Load Configuration File
cfg = load_config(f'./config.yaml')

# Set Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basis of the data directory
base_path = './data/open/'

# To reproduce our final submission, load every reference before inference
if reproduce:
    base_id = 17
    ref_dict = pickle.load(open(f'./history/ref_dict_id_{base_id}_multi.pickle', 'rb'))

    total_fake = ref_dict['fake']
    total_real = ref_dict['real']

# If you don't want to reproduce our final submission, it loads a random reference sampler
else:
    (_, _), fake_file_list, real_file_list = load_path_list(base_path)
    fake_dataset = ReferenceDataset(fake_file_list, base_path, shuffle = True)
    real_dataset = ReferenceDataset(real_file_list, base_path, shuffle = True)

    # Not Actually Used. It's like Vestigial Organ.
    if 'seed.pickle' not in os.listdir(f'./history/'):
        seed = [np.random.randint(0, 1000) for _ in range(len(fake_dataset))]
        pickle.dump(seed, open(f'./history/seed.pickle', 'wb'))

    else:
        seed = pickle.load(open(f'./history/seed.pickle', 'rb'))

    total_fake, total_real = list(), list()

# Load every parameter filenames
parameters = os.listdir(f'./ensemble/')

# Function that define model and load parameters
def define_model(parameters):
    model = SiamMetricNetworks(cfg['INPUT_DIM'], cfg['HIDDEN_DIM'], cfg['OUTPUT_DIM']).to(device)
    model.load_state_dict(torch.load(f'./ensemble/{parameters}', map_location = device))

    return model

# Define models for each parameters file
models = [define_model(parameter) for parameter in parameters]

# Load every filename for testing
test_file_list = load_path_list(base_path, test = True)

# Set Custom Inference Dataset
test_dataset = InferenceDataset(test_file_list, base_path)

# You should keep 'shuffle' 'False' to make submission file
test_loader = DataLoader(test_dataset, batch_size = cfg['BATCH_SIZE'], shuffle = False)

# Buffer for save every prediction
total_preds = list()

# To reproduce our final submission file
if reproduce:
    for idx, (model, model_fake, model_real) in enumerate(zip(models, total_fake, total_real)):

        # Load previously-loaded reference files and make predictions.

        total_pred = list()
        for anchor, ref_fake_list, ref_real_list in tqdm(zip(test_loader, model_fake, model_real), desc = f'[Reproduce] Inferencing [{idx + 1} / {len(models)}]...'):
            anchor = anchor.to(device)

            ref_fake_list = [batch.to(device) for batch in ref_fake_list]
            ref_real_list = [batch.to(device) for batch in ref_real_list]

            pred_fake, pred_real = list(), list()
            for batch_fake, batch_real in zip(ref_fake_list, ref_real_list):
                _pred_fake, _pred_real = model(anchor, batch_fake, batch_real)

                if apply_threshold:
                    _pred_fake = torch.where(_pred_fake > thres, 1.0, 0.0)
                    _pred_real = torch.where(_pred_real > thres, 1.0, 0.0)

                pred_fake.append(_pred_fake.unsqueeze(-1))
                pred_real.append(_pred_real.unsqueeze(-1))

            pred_fake = torch.cat(pred_fake, dim=-1).mean(dim=-1)
            pred_real = torch.cat(pred_real, dim=-1).mean(dim=-1)

            pred = torch.cat([pred_fake, pred_real], dim = 1).cpu().detach().numpy()

            total_pred.append(pred)

        total_pred = np.concatenate(total_pred, axis = 0)
        total_preds.append(total_pred)

# To make a new prediction with new reference combinations
else:
    for idx, model in enumerate(models):

        model_fake, model_real = list(), list()
        total_pred = list()
        for anchor in tqdm(test_loader, desc = f'Inferencing [{idx + 1} / {len(models)}]...'):
            anchor = anchor.to(device)

            ref_fake_list = [[] for _ in range(n_heads)]
            ref_real_list = [[] for _ in range(n_heads)]

            # Resample references for prediction
            for i in range(anchor.shape[0]):
                for head_id in range(n_heads):
                    ref_fake_list[head_id].append(fake_dataset[i].unsqueeze(0))
                    ref_real_list[head_id].append(real_dataset[i].unsqueeze(0))

            ref_fake_list = [torch.cat(batch, dim = 0) for batch in ref_fake_list]
            ref_real_list = [torch.cat(batch, dim = 0) for batch in ref_real_list]

            model_fake.append(ref_fake_list)
            model_real.append(ref_real_list)

            ref_fake_list = [batch.to(device) for batch in ref_fake_list]
            ref_real_list = [batch.to(device) for batch in ref_real_list]

            pred_fake, pred_real = list(), list()
            for batch_fake, batch_real in zip(ref_fake_list, ref_real_list):
                _pred_fake, _pred_real = model(anchor, batch_fake, batch_real)

                if apply_threshold:
                    _pred_fake = torch.where(_pred_fake > thres, 1.0, 0.0)
                    _pred_real = torch.where(_pred_real > thres, 1.0, 0.0)

                pred_fake.append(_pred_fake.unsqueeze(-1))
                pred_real.append(_pred_real.unsqueeze(-1))

            pred_fake = torch.cat(pred_fake, dim = -1).mean(dim = -1)
            pred_real = torch.cat(pred_real, dim = -1).mean(dim = -1)

            pred = torch.cat([pred_fake, pred_real], dim = 1).cpu().detach().numpy()

            total_pred.append(pred)

        total_pred = np.concatenate(total_pred, axis = 0)
        total_preds.append(total_pred)

        total_fake.append(model_fake)
        total_real.append(model_real)

# Soft Voting, Averaging every prediction
total_pred_ensemble = np.array(total_preds).mean(axis = 0)

# If the predictions were made with totally new reference combinations, save the combinations for further reproduction
if not reproduce:
    ref_dict = {
        'fake': total_fake,
        'real': total_real
    }
    pickle.dump(ref_dict, open(f'./history/ref_dict_id_{inference_id}_multi.pickle', 'wb'))

# Load the sample submission file
test_df = pd.read_csv(base_path + 'sample_submission.csv')

# Insert our predictions
test_df['fake'] = total_pred_ensemble[:, 0]
test_df['real'] = total_pred_ensemble[:, 1]

# Save the predictions for submission
if reproduce:
    test_df.to_csv(f'./submission/Submission_prediction_id_{inference_id}_head_{n_heads}_reproduce_base_{base_id}_ensemble.csv', index=False)
else:
    test_df.to_csv(f'./submission/Submission_prediction_id_{inference_id}_head_{n_heads}_ensemble.csv', index = False)