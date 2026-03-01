import numpy as np
import argparse
import torch
import random
import os
import json
from src.create_dataset import create_ai_hub_train_data_info, run_preprocessing, extract_features
from src.train import model_train, predict_emotion

def set_seed(seed=2025):
    """실험 결과의 재현성을 위해 난수 시드를 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"시드가 {seed}로 고정되었습니다.")


def main():
    train_data = None
    X = None
    Y = None

    parser = argparse.ArgumentParser()
    # 실행 옵션 설정
    parser.add_argument('--set_config', type=str, default='config.json', help='설정 파일 경로')
    parser.add_argument('--create_train_data', action='store_true', help='학습 데이터 생성')
    parser.add_argument('--start_train', action='store_true', help='model 학습 시작')
    parser.add_argument('--predict', type=str, help='예측용 오디오 파일 경로 입력')
    args = parser.parse_args()

    # 설정값 로드
    with open(args.set_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    set_seed(seed=config['seed'])

    # 데이터 생성 단계
    if args.create_train_data:
        train_data = create_ai_hub_train_data_info(config)
        X, Y = run_preprocessing(train_data.path.values, train_data.labels.values, config['save_data_path'])

    # 학습 및 평가 로직
    if args.start_train:
        if args.create_train_data:
            model_train(X, Y, config)
        else:
            try:
                X = np.load(config['save_data_path'] + 'Train_features.npy')
                Y = np.load(config['save_data_path'] + 'Train_labels.npy')
            except:
                print('none data')
                return None
            model_train(X, Y, config)

    if args.predict:
        feature = extract_features(args.predict, sr=16000, duration=20.0)
        logits = predict_emotion(feature)
        return logits
    return None

if __name__ == '__main__':
    logits = main()