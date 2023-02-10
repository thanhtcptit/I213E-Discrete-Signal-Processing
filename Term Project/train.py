import warnings
warnings.filterwarnings('ignore')

import random

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import librosa
import skimage.io as io

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def scale(x, _min=0.0, _max=1.0):
    std = (x - x.min()) / (x.max() - x.min())
    return std * (_max - _min) + _min


def preprocess(file_path, sr=None, start_bin=0, sample_duration=30,
               n_fft=2048, hop_length=512, n_mels=128, max_frames=1293,
               scale_int8=False):
    wave_data, sr = librosa.load(file_path, sr=sr, dtype=np.float32)

    start_sample_bin = start_bin * sr
    end_sample_bin = start_sample_bin + sample_duration * sr + 1
    max_length = sr * sample_duration

    wave_data = wave_data[start_sample_bin: end_sample_bin]
    wave_length, = wave_data.shape
    d = max_length - wave_length
    if d > 0:
        wave_data = np.hstack((wave_data, np.zeros((d), dtype=wave_data.dtype)))
    elif d < 0:
        wave_data = wave_data[:d]

    wave_data = librosa.util.normalize(wave_data)
    mel_spec = librosa.feature.melspectrogram(
        wave_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec = librosa.power_to_db(mel_spec)
    mel_spec = np.transpose(mel_spec)

    d = max_frames - mel_spec.shape[0]
    if d > 0:
        mel_spec = np.vstack((mel_spec, np.zeros((d, mel_spec.shape[1]))))
    elif d < 0:
        mel_spec = mel_spec[:max_frames, :]
    if scale_int8:
        mel_spec = scale(mel_spec, 0, 255).astype(np.uint8)
    return mel_spec


def transform_dataset(data_dir, output_dir, override=False):
    class_list = sorted(os.listdir(data_dir))
    X, y = [], []
    for i, g in enumerate(class_list):
        genre_dir = data_dir / g
        out_dir = output_dir / g
        audio_files = sorted(os.listdir(genre_dir))
        for f in audio_files:
            file_path = genre_dir / f
            save_path = out_dir / str(os.path.splitext(f)[0] + ".npy")
            if not os.path.exists(save_path) or (os.path.exists(save_path) and override):
                out_dir.mkdir(exist_ok=True, parents=True)
                mel_spec = preprocess(file_path)
                np.save(save_path, mel_spec)
                # image_path = out_dir / str(os.path.splitext(f)[0] + ".png")
                # io.imsave(image_path, img)

            X.append(save_path)
            y.append(i)
    return X, y


class MelSpecDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, convert_melspec=False, transform=None):
        self.X = X
        self.y = y
        self.convert_melspec = convert_melspec
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.convert_melspec:
            mel_spec = preprocess(self.X[idx])
        else:
            # mel_spec = (io.imread(self.X[idx]) / 255.).astype(np.float32)
            mel_spec = np.load(self.X[idx]).astype(np.float32)
        label = self.y[idx]
        if self.transform:
            mel_spec = self.transform(mel_spec)
        return mel_spec, label


class MelSpecNet(nn.Module):
    def __init__(self, n_classes, kernel_size=3, pool_size=6, dropout=0.8):
        super().__init__()

        self.n_channels = [1, 4, 8, 16, 16]
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(self.n_channels[i])for i in range(len(self.n_channels) - 1)]
            )
        self.conv = nn.ModuleList(
            [nn.Conv2d(self.n_channels[i], self.n_channels[i + 1], kernel_size)
             for i in range(len(self.n_channels) - 1)]
            )
        self.act = nn.ELU()
        self.pool = nn.MaxPool2d(pool_size)
        self.dropconv = nn.ModuleList(
            [nn.Dropout2d(0.) for _ in range(len(self.n_channels) - 1)]
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.LazyLinear(n_classes)
        )

    def forward(self, x):
        for i in range(len(self.n_channels) - 1):
            x = self.pool(self.dropconv[i](self.act(self.conv[i](self.bn[i](x)))))
        # x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def evaluate(model, loader, class_list, device):
    correct_pred = {c: 0 for c in class_list}
    total_pred = {c: 0 for c in class_list}

    model.eval()
    with torch.no_grad():
        for (x, y) in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = torch.max(out, 1)
            for label, pred in zip(y, preds):
                if label == pred:
                    correct_pred[class_list[label]] += 1
                total_pred[class_list[label]] += 1
    total_acc = []
    for c, correct in correct_pred.items():
        acc = 100 * float(correct) / total_pred[c]
        total_acc.append(acc)
    return np.mean(total_acc), total_acc


def train(n_epochs, batch_size, lr, optim_, kernel_size, pool_size, dropout,
          convert_melspec_online=False, log=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)

    DATA_DIR = Path("/home/s2210421/datasets/gtzan/genres")
    MELSPEC_DIR = Path("/home/s2210421/datasets/gtzan/melspec")

    genre_list = sorted(os.listdir(DATA_DIR))
    num_classes = len(genre_list)

    if convert_melspec_online:
        X, y = [], []
        for i, g in enumerate(genre_list):
            genre_dir = DATA_DIR / g
            audio_files = sorted(os.listdir(genre_dir))
            for f in audio_files:
                file_path = genre_dir / f
                X.append(file_path)
                y.append(i)
    else:
        X, y = transform_dataset(DATA_DIR, MELSPEC_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    if log:
        print(f"Train: {len(X_train)} - Test: {len(X_test)}")

    transform_set = [transforms.ToTensor()]
    transform = transforms.Compose(transform_set)
    train_dataset = MelSpecDataset(X_train, y_train, convert_melspec_online, transform)
    test_dataset = MelSpecDataset(X_test, y_test, convert_melspec_online, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4)

    model = MelSpecNet(num_classes, kernel_size=kernel_size, pool_size=pool_size,
                       dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    if optim_ == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    best_score = 0
    log_step = 10
    for epoch in range(n_epochs):
        running_loss = 0
        if log:
            pbar = tqdm(enumerate(train_loader, 1))
        else:
            pbar = enumerate(train_loader, 1)
        for i, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % log_step == 0:
                test_acc, _ = evaluate(model, test_loader, genre_list, device)
                if test_acc > best_score:
                    best_score = test_acc
                    torch.save(model, "cnn.pt")
                if log:
                    pbar.set_postfix({"Epoch": epoch + 1, "step": i, "loss": running_loss / log_step,
                                      "test_acc": test_acc})
                running_loss = 0
                model.train()

    return best_score


def test(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)

    DATA_DIR = Path("/home/s2210421/datasets/gtzan/genres")
    MELSPEC_DIR = Path("/home/s2210421/datasets/gtzan/melspec")

    genre_list = sorted(os.listdir(DATA_DIR))

    X, y = transform_dataset(DATA_DIR, MELSPEC_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    transform_set = [transforms.ToTensor()]
    transform = transforms.Compose(transform_set)
    test_dataset = MelSpecDataset(X_test, y_test, False, transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, num_workers=4)
    
    model = torch.load(model_path, map_location="cpu").to(device)
    avg_acc, genre_acc = evaluate(model, test_loader, genre_list, device)
    for g, a in zip(genre_list, genre_acc):
        print(f"{g}: {a}")
    print(f"Average accuracy: {avg_acc}")


if __name__ == "__main__":
    # train(*sys.argv[1:])
    # acc = train(1000, 32, 0.004, "adam", 3, (4, 2), 0.8)
    # print(acc)

    test("best_cnn.pt")
