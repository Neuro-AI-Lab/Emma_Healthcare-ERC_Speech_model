from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.model import EmotionCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_train(X, Y, config):
    print('Data Split ...')
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=2025)
    train_loader = DataLoader(TensorDataset(torch.Tensor(x_train).unsqueeze(1), torch.LongTensor(y_train)),
                              batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(x_val).unsqueeze(1), torch.LongTensor(y_val)),
                            batch_size=config['batch_size'])
    print('Done')
    print('Using device:', device)
    model = EmotionCNN(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        # --- 학습 단계 ---
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for bx, by in train_pbar:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # 진행바 오른쪽에 현재 배치 Loss 표시
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- 검증 단계 ---
        model.eval()
        val_correct = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=False)

        with torch.no_grad():
            for bx, by in val_pbar:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                preds = outputs.argmax(1)
                val_correct += (preds == by).sum().item()

        val_acc = val_correct / len(x_val)

        # 에폭 종료 후 결과 출력
        print(f"\n=> Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 최고 정확도 갱신 시 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './Result/best_model.pth')
            print(f"에폭 {epoch}: 최고 정확도 갱신 ({best_acc:.4f}) -> 저장 완료")


def predict_emotion(feature):
    # 1. 파일 전처리

    input_tensor = torch.Tensor(feature).view(1, 1, -1).to(device)

    # 2. 모델 가중치 로드
    test_model = EmotionCNN(input_dim=len(feature)).to(device)
    test_model.load_state_dict(torch.load('./Result/best_model.pth'))
    test_model.eval()

    # 3. 예측
    with torch.no_grad():
        pred = test_model(input_tensor).to('cpu').numpy()
    return pred