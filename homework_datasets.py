import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(n=100, source='random'):
    if source == 'random':
        X = torch.rand(n, 2)
        w = torch.tensor([2.0, -3.0])
        b = 0.5
        logits = X @ w + b
        y = (logits > 0).float().unsqueeze(1)
        return X, y
    elif source == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')


class IrisDataset(ClassificationDataset):
    """
    Загрузка датасета Iris из CSV с предобработкой и опциональным feature engineering.
    :param csv_path: путь к CSV-файлу (должен содержать все признаки и колонку 'target')
    :param featureEngineering: если True, генерируются полиномиальные, взаимодействия и статистические признаки
    """

    def __init__(self, csv_path: str, featureEngineering: bool = False):
        # 1. Загрузка из CSV
        df = pd.read_csv(csv_path)

        if 'Species' not in df.columns:
            raise ValueError("CSV должен содержать колонку 'Species' с метками классов")

        # меняю названия классов на цифры
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Species'])
        X = df.drop(columns=['Species'])

        # выполняю ohe для категориальных признаков
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            ohe = OneHotEncoder(sparse=False, drop='if_binary')
            X_cat = ohe.fit_transform(X[cat_cols])
        else:
            X_cat = np.empty((len(X), 0))

        # стандартизация
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X[num_cols])
        else:
            X_num = np.empty((len(X), 0))

        X_proc = np.hstack([X_num, X_cat])

        # 4. Feature engineering
        if featureEngineering:
            # полиномиальные
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_poly = poly.fit_transform(X_proc)

            # статистические признаки
            row_mean = X_proc.mean(axis=1, keepdims=True)
            row_var = X_proc.var(axis=1, keepdims=True)

            X_proc = np.hstack([X_proc, X_poly, row_mean, row_var])

        X_tensor = torch.tensor(X_proc, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)  # предполагаем, что target — целочисленный код класса

        # передаём в базовый класс
        super().__init__(X_tensor, y_tensor)
        self.featureEngineering = featureEngineering
