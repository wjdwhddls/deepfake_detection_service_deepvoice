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
import librosa
import soundfile as sf  # WAV 파일 저장을 위해 soundfile을 임포트합니다.
import shutil
import json

# 모델 경로 및 데이터 경로 설정  
model_path = '/Users/gyubeom/Desktop/Final Project/deepfake_detection_service_deepvoice/WeSpeaker_ResNet221'  
base_path = './data/open/'  

# WeSpeaker 모델 로드  
embedding_model = ws.load_model_local(model_path)  
embedding_model.set_device('mps')  # MPS 디바이스 사용  

# 진짜/가짜 판단을 위한 모델 로드  
from model import SiamMetricNetworks  
cfg = load_config('/Users/gyubeom/Desktop/Final Project/deepfake_detection_service_deepvoice/config.yaml')  
classifier_model = SiamMetricNetworks(cfg['INPUT_DIM'], cfg['HIDDEN_DIM'], cfg['OUTPUT_DIM'])  
classifier_model.to('mps')  
classifier_model.load_state_dict(torch.load('/Users/gyubeom/Desktop/Final Project/deepfake_detection_service_deepvoice/history/model/Epoch_33_best_model.pth', map_location='mps'))  
classifier_model.eval()  

# 데이터 경로 로드  
(anchor_path_list, anchor_label_list), fake_file_list, real_file_list = load_path_list(base_path)  
path_dict = pickle.load(open('/Users/gyubeom/Desktop/Final Project/deepfake_detection_service_deepvoice/history/train_valid_test.pickle', 'rb'))  

# 데이터셋 및 로더  
anchor_dataset = AnchorDataset(anchor_path_list, anchor_label_list, base_path)  
fake_dataset = ReferenceDataset(fake_file_list, base_path, shuffle=True)  
real_dataset = ReferenceDataset(real_file_list, base_path, shuffle=True)  
anchor_loader = DataLoader(anchor_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, drop_last=False)  

def convert_m4a_to_ogg(file_path):  
    audio = AudioSegment.from_file(file_path, format='m4a')  
    ogg_file_path = file_path.replace('.m4a', '.ogg')  
    audio.export(ogg_file_path, format='ogg')  
    return ogg_file_path  

def extract_embedding(file_path):  
    embedding = embedding_model.extract_embedding(file_path)  
    return embedding  

def get_reference_batches(batch_size):  
    fake_samples = [fake_dataset[i] for i in np.random.choice(len(fake_dataset), batch_size)]  
    real_samples = [real_dataset[i] for i in np.random.choice(len(real_dataset), batch_size)]  
    ref_fake = torch.stack(fake_samples, dim=0).to('mps')  
    ref_real = torch.stack(real_samples, dim=0).to('mps')  
    return ref_fake, ref_real  

def predict(file_path):  
    if file_path.endswith('.m4a'):  
        ogg_file_path = convert_m4a_to_ogg(file_path)  
        if ogg_file_path is None:  
            return  
        file_to_process = ogg_file_path  
    elif file_path.endswith('.wav'):  
        file_to_process = file_path  
    else:  
        print("지원하지 않는 파일 형식입니다.")  
        return  

    segment_length_sec = 2  
    output_dir = os.path.join(base_path, "segments")  
    split_audio(file_to_process, output_dir, segment_length_sec)  

    segment_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.wav')]  
    true_count = 0  
    fake_count = 0  
    final_real_prob = 0  
    final_fake_prob = 0  
    
    for segment_file in segment_files:  
        embedding = extract_embedding(segment_file)  
        embedding_tensor = torch.tensor(embedding).unsqueeze(0).to('mps')  
        ref_fake, ref_real = get_reference_batches(cfg['BATCH_SIZE'])  

        with torch.no_grad():  
            pred_fake, pred_real = classifier_model(embedding_tensor, ref_fake, ref_real)  

        pred_fake_prob = torch.sigmoid(pred_fake).cpu().numpy()[0][0] * 100  
        pred_real_prob = torch.sigmoid(pred_real).cpu().numpy()[0][0] * 100  

        if pred_real_prob > pred_fake_prob:  
            true_count += 1  
            final_real_prob += pred_real_prob  
        else:  
            fake_count += 1  
            final_fake_prob += pred_fake_prob  

    if true_count > fake_count:  
        final_result = "진짜입니다."  
        average_real_prob = final_real_prob / true_count if true_count > 0 else 0  
        average_fake_prob = 100 - average_real_prob  
    else:  
        final_result = "가짜입니다."  
        average_fake_prob = final_fake_prob / fake_count if fake_count > 0 else 0  

    # 위험도 판별  
    if average_fake_prob < 30:  
        risk_level = "안전"  
    elif average_fake_prob < 60:  
        risk_level = "경고"  
    else:  
        risk_level = "위험"  

    result_json = {  
        "result": final_result,  
        "average_fake_prob": average_fake_prob,  
        "risk_level": risk_level  
    }  

    # 최종 JSON 결과 출력  
    print(json.dumps(result_json))  

def get_latest_file(directory: str) -> str:  
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]  
    if not files:  
        return None  
    latest_file = max(files, key=os.path.getctime)  
    return latest_file  

def split_audio(file_path, output_dir, segment_length_sec=2):  
    print(f"Splitting audio file: {file_path} into segments of {segment_length_sec} seconds.")  
    audio, sample_rate = librosa.load(file_path, sr=None)  
    segment_length_samples = int(segment_length_sec * sample_rate)  

    if os.path.exists(output_dir):  
        shutil.rmtree(output_dir)  
    os.makedirs(output_dir)  

    total_samples = len(audio)  
    segments_count = total_samples // segment_length_samples  

    for i in range(segments_count):  
        start_sample = i * segment_length_samples  
        end_sample = start_sample + segment_length_samples  
        segment = audio[start_sample:end_sample]  
        output_file = os.path.join(output_dir, f"segment_{i + 1}.wav")  
        sf.write(output_file, segment, sample_rate)  

# 실행부  
directory_path = '/Users/gyubeom/Desktop/Final Project/deepfake_detection_service_deepvoice/data_example'  
latest_file_path = get_latest_file(directory_path)  

if latest_file_path:  
    print(f"가장 최근 파일: {latest_file_path}")  
    predict(latest_file_path)  
else:  
    print("폴더에 파일이 없습니다.")