import torch  
import os  
import numpy as np  
from pydub import AudioSegment  
from utils import load_config  
import wespeaker as ws  
from tqdm.auto import tqdm  
from MainDataset import *  
import pickle  
from torch.utils.data import DataLoader  

# 모델 경로 및 데이터 경로 설정  
model_path = '/Users/jongin/deepfake_detection_service_deepvoice/WeSpeaker_ResNet221/'  
base_path = './data/open/'  

# WeSpeaker 모델 로드  
embedding_model = ws.load_model_local(model_path)  

# Set device (MPS for M1/M2/M3)  
embedding_model.set_device('mps')  # Speaker 클래스의 set_gpu 메서드 사용  

# 진짜/가짜 판단을 위한 모델 로드  
from model import SiamMetricNetworks  # 모델 클래스 임포트  
cfg = load_config('/Users/jongin/deepfake_detection_service_deepvoice/config.yaml')  # 설정 로드  
classifier_model = SiamMetricNetworks(cfg['INPUT_DIM'], cfg['HIDDEN_DIM'], cfg['OUTPUT_DIM'])  
classifier_model.to('mps')  # MPS로 모델 이동  

# 모델 로드: Epoch_32_best_model.pth 사용  
classifier_model.load_state_dict(torch.load('/Users/jongin/deepfake_detection_service_deepvoice/history/model/Epoch_32_best_model.pth', map_location='mps'))  
classifier_model.eval()  # 평가 모드로 전환  

# 데이터 경로 로드  
base_path = '/Users/jongin/deepfake_detection_service_deepvoice/data/open/'  
(anchor_path_list, anchor_label_list), fake_file_list, real_file_list = load_path_list(base_path)  

path_dict = pickle.load(open(f'/Users/jongin/deepfake_detection_service_deepvoice/history/train_valid_test.pickle', 'rb'))  

# 데이터셋을 설정  
anchor_dataset = AnchorDataset(anchor_path_list, anchor_label_list, base_path)  
fake_dataset = ReferenceDataset(fake_file_list, base_path, shuffle=True)  
real_dataset = ReferenceDataset(real_file_list, base_path, shuffle=True)  

# 데이터 로더  
anchor_loader = DataLoader(anchor_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, drop_last=False)  

def convert_m4a_to_ogg(file_path):  
    # .m4a 파일을 .ogg 파일로 변환  
    audio = AudioSegment.from_file(file_path, format='m4a')  
    ogg_file_path = file_path.replace('.m4a', '.ogg')  
    audio.export(ogg_file_path, format='ogg')  
    return ogg_file_path  

def extract_embedding(file_path):  
    # .ogg 파일을 로드  
    embedding = embedding_model.extract_embedding(file_path)  # 파일 경로를 전달  
    return embedding  

def get_reference_batches(batch_size):  
    # 가짜 및 진짜 배치 가져오기  
    fake_samples = [fake_dataset[i] for i in np.random.choice(len(fake_dataset), batch_size)]  
    real_samples = [real_dataset[i] for i in np.random.choice(len(real_dataset), batch_size)]  
    
    ref_fake = torch.stack(fake_samples, dim=0).to('mps')  
    ref_real = torch.stack(real_samples, dim=0).to('mps')  
   
    return ref_fake, ref_real  

def predict(file_path):  
    # 파일 확장자에 따라 처리  
    if file_path.endswith('.m4a'):  
        ogg_file_path = convert_m4a_to_ogg(file_path)  
        if ogg_file_path is None:  
            return  # 변환 실패 시 함수 종료  
        file_to_process = ogg_file_path  
    elif file_path.endswith('.wav'):  
        file_to_process = file_path  # .wav 파일은 변환할 필요 없음  
    else:  
        print("지원하지 않는 파일 형식입니다.")  
        return   
    
    # 임베딩 생성  
    embedding = extract_embedding(file_to_process)  

    # 임베딩을 텐서로 변환하고 배치 차원 추가  
    embedding_tensor = torch.tensor(embedding).unsqueeze(0).to('mps')  # Ensure embedding is on the correct device  

    # Pre-Sampling: 가짜 및 진짜 배치 가져오기  
    ref_fake, ref_real = get_reference_batches(cfg['BATCH_SIZE'])  

    # 예측 수행  
    with torch.no_grad():  
        pred_fake, pred_real = classifier_model(embedding_tensor, ref_fake, ref_real)  # 모델에 임베딩 입력  

    # 예측 확률을 사용  
    pred_fake_prob = torch.sigmoid(pred_fake).cpu().numpy()[0][0]  # 가짜 확률  
    pred_real_prob = torch.sigmoid(pred_real).cpu().numpy()[0][0]  # 진짜 확률  

    # 결과 출력  
    if pred_real_prob > pred_fake_prob:  # 진짜 확률이 가짜 확률보다 높으면  
        result = "진짜입니다."  
        probability = pred_real_prob * 100  
    else:  # 가짜 확률이 진짜 확률보다 높으면  
        result = "가짜입니다."  
        probability = pred_fake_prob * 100  

    print(f"이 파일은 {result} 확률: {probability:.2f}%")   

def get_latest_file(directory: str) -> str:  
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]  
    if not files:  
        return None  
    latest_file = max(files, key=os.path.getctime)  # 가장 최근 생성된 파일 찾기  
    return latest_file  

# 최신 파일 찾기  
directory_path = '/Users/jongin/deepfake_detection_service_deepvoice/data_example'  
latest_file_path = get_latest_file(directory_path)  

if latest_file_path:  
    print(f"가장 최근 파일: {latest_file_path}")  
    predict(latest_file_path)  # 예측 수행  
else:  
    print("폴더에 파일이 없습니다.")  