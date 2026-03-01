import os
import pandas as pd
import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def create_ai_hub_train_data_info(config):
    """AI HUB 학습 데이터 파일을 정리합니다."""

    # 데이터 경로 설정 및 불러오기
    subject_path = config['train_data_path'] + 'voice/'
    label_path = config['train_data_path'] + 'label/'
    subject_list = os.listdir(subject_path)
    label_list = os.listdir(label_path)
    save_path = config['save_data_path'] + 'train_data.tsv'
    result = []
    for label in label_list:
        print(f'{label} 데이터 전처리 중...')
        temp_df = pd.read_csv(label_path + label, encoding='cp949')
        temp_df['상황'] = temp_df['상황'].replace({'anger':'angry', 'fear':'fearful', 'sadness':'sad', 'happiness':'happy'})
        for i in range(len(temp_df)):
            subject_id = temp_df.iloc[i]['wav_id'] + '.wav'
            temp_label = temp_df.iloc[i]['상황']
            if subject_id in subject_list:
                result.append({'labels': config['emotion_map'][temp_label], 'path':subject_path+subject_id})

    data_info_df = pd.DataFrame(result)
    data_info_df.to_csv(save_path, sep='\t', index=False)
    print(f'Data info {save_path}에 저장되었습니다.')
    return data_info_df

def extract_features(path, sr=16000, duration=20.0):
    try:
        data, _ = librosa.load(path, sr=sr, duration=duration, offset=0)
        target_length = int(sr * duration)

        if len(data) < target_length:
            data = np.pad(data, (0, target_length - len(data)), 'constant')

        zcr = np.squeeze(librosa.feature.zero_crossing_rate(y=data))
        rmse = np.squeeze(librosa.feature.rms(y=data))
        mfcc = np.ravel(librosa.feature.mfcc(y=data, sr=sr).T)

        return np.hstack((zcr, rmse, mfcc))
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_preprocessing(paths, labels, save_path):
    print("전처리 시작")
    features = Parallel(n_jobs=-1)(delayed(extract_features)(p) for p in tqdm(paths))

    X = np.array([f for f in features if f is not None])
    Y = np.array([l for f, l in zip(features, labels) if f is not None])

    np.save(save_path+'Train_features.npy', X)
    np.save(save_path+'Train_labels.npy', Y)
    print(f"전처리 완료! 특징 차원: {X.shape[1]}")
    return X, Y