# coding:utf8
import torch
import torch.nn as nn
import numpy as np


class MultiClassificationModel(nn.Module):
    def __init__(self, input_size=5, num_classes=5):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)  # logits


def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y


def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)


def evaluate(model, test_sample_num=1000):
    model.eval()
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()
    print(f"正确率: {acc:.4f}")
    return acc


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    learning_rate = 0.005

    model = MultiClassificationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    for epoch in range(epoch_num):
        model.train()
        train_x, train_y = build_dataset(train_sample)  # 每轮重采样
        losses = []

        for i in range(train_sample // batch_size):
            x = train_x[i * batch_size:(i + 1) * batch_size]
            y = train_y[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        print(f"========= 第{epoch+1}轮平均loss: {avg_loss:.6f}")
        acc = evaluate(model)
        log.append([acc, avg_loss])

    torch.save(model.state_dict(), "model.pt")
    print(log)


def predict(model_path, input_vec):
    model = MultiClassificationModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x = torch.tensor(input_vec, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    for vec, pred, prob in zip(input_vec, preds, probs):
        print(f"输入: {vec}, 预测类别: {pred.item()}, 概率值: {prob.numpy()}")


if __name__ == "__main__":
    main()
    test_vec = [
        [0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
        [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
        [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
        [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894],
    ]
    predict("model.pt", test_vec)
