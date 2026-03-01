# Speech 기반 감정 분류 플랫폼 (Speech-Emotion_Model)

본 프로젝트는 음성 데이터를 활용하여 7가지 감정으로 분류하는 통합 파이프라인입니다. 시계열 데이터의 특징을 추출하고, 전이 학습(Transfer Learning)을 통해 타겟 데이터셋에 최적화된 성능을 도출하도록 설계되었습니다.

---
- **참여 기관**: 광운대학교 신경공학 및 인공지능 연구실 (NeuroAI Lab)
- **개발자**:
  - 이준영 연구원
    - leejykw2025@kw.ac.kr
  - 이동혁 연구원
  - 김대현 연구원
---

## 1. 감정 분류 클래스 (7개)
데이터셋은 다음 7가지 감정을 분류하도록 구성되어 있습니다:
- `happy` (0), `neutral` (1), `fearful` (2), `disgust` (3), `surprise` (4), `sad` (5), `angry` (6)

## 2. Configuration 설정 (config.json)
프로젝트의 주요 파라미터 및 경로는 `config.json`에서 관리됩니다:

| Parameter         | Description               |
|:------------------|:--------------------------|
| `train_weight`    | 학습이 끝난 모델 가중치             |
| `train_data_path` | 전이 학습용 원천 데이터셋 경로         |
| `save_data_path`  | 데이터셋이 저장될 경로 |
| `batch_size`      | 학습 시 사용할 배치 크기            |
| `learning_rate`   | 모델 최적화를 위한 학습률            |
| `epochs`          | 학습 반복 횟수                  |

## 3. Main.py Argument 상세 설명

`main.py`는 명령행 인자를 통해 전처리, 학습, 평가를 제어합니다.

| Argument                   | Type | Default | 상세 기능 설명                                 |
|:---------------------------| :--- | :--- |:-----------------------------------------|
| `--set_config`             | `str` | `config.json` | 설정(경로, 시드, 파라미터) 파일 지정                   |
| `--create_train_data`      | `flag` | - | AI HUB speech 데이터를 전처리하여 numpy 파일로 저장    |
| `--start_train` | `flag` | - | 모델을 학습하고 최적 모델을 저장 |
| `--predict`                | `str` | - | 특정 `.wav` 파일의 감정 상태를 즉시 예측               |

## 4. 학습 알고리즘 및 최적 모델 저장 (src/train.py)

프로젝트는 CNN기반의 Supervised-learning을 활용한 모델입니다.:

### Stage 1: Training (모델 학습)
* **목적**: Train data를 모델에 학습시킴
* **결과물**: `./Result/best_model.pth`에 가중치 저장.

## 5. 실행 가이드 (Usage)
- **예측 시**: 별도의 학습 없이 결과만 확인하고 싶으시다면 `Step 2`를 바로 실행하시면 됩니다.
### Step 1: 필요한 라이브러리 설치
- 현재 본 모델은 `Python 3.10`을 사용하고 있습니다.
- 사전에 필요한 라이브러리는 `requirements.txt`에 저장해 두었습니다.
- 최종 학습 모델 가중치가 필요합니다. 아래의 구글드라이브 Zip파일을 해제하여 Result 폴더에 넣어주세요.
- https://drive.google.com/file/d/1vowvLQreTQ_Z07CJItbyIlfE64NYCSfU/view?usp=drive_link
```bash
pip install -r requirements.txt
```
### Step 1: 데이터 전처리 및 모델 학습
- 이미 train data가 Dataset 폴더에 들어있다면 --create_train_data 인자는 제외후 사용하시면 됩니다.
```bash
python main.py --create_train_data --start_train
```

### Step 2: 실시간 감정 예측
- 학습된 모델을 사용하여 특정 음성 데이터의 감정을 즉시 분석합니다.

**음성**
```bash
python main.py --predict "음성 데이터 경로"
```
## 6. 결과 산출물 (Results)
- 최적 가중치: ./Result/best_model.pth
