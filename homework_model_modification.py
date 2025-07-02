import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework_datasets import make_regression_data, RegressionDataset, make_classification_data, ClassificationDataset, IrisDataset
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LinearRegression(nn.Module):
    def __init__(
            self,
            in_features: int,
            regularization: str = 'l1',
            l1_lambda: float = 0.01,
            l2_lambda: float = 0.01
    ):
        """
        Функция инициализации класса линейной регрессии c early stopping и регуляризацией
        :param in_features: Размер входного признакового пространства
        :param regularization: Тип регуляризации ('l1' или 'l2'). По умолчанию 'l1'.
        :param l1_lambda: Коэффициент для L1-регуляризации. По умолчанию 0.01.
        :param l2_lambda: Коэффициент для L2-регуляризации. По умолчанию 0.01.
        """
        super().__init__()
        self.regularization = regularization
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.linear = nn.Linear(in_features, 1)
        # Лучшая потеря, достигнутая на валидационной выборке
        self.best_loss = float('inf')
        # Лучшие веса, достигнутые на валидационной выборке
        self.best_state_dict = None

    def forward(self, x):
        """
        Прямой проход модели
        :param x: Входные данные для прямого прохода
        :return: Результаты предсказания модели
        """
        return self.linear(x)

    def regularization_loss(self) -> torch.Tensor:
        """Вычисляет регуляризационную потерю (L1/L2)"""
        reg_loss = torch.tensor(0.)
        for param in self.parameters():
            if self.regularization == 'l1':
                reg_loss += self.l1_lambda * torch.abs(param).sum()
            elif self.regularization == 'l2':
                reg_loss += self.l2_lambda * torch.pow(param, 2).sum()
        return reg_loss

    def train_model(
            self,
            dataloader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            epochs: int = 100,
            patience: int = 5,
            min_delta: float = 0.001,
            verbose: bool = True
    ) -> None:
        """
        Обучение модели с регуляризацией и ранней остановкой
        :param dataloader: DataLoader с обучающими данными
        :param criterion: Функция потерь (nn.MSELoss())
        :param optimizer: Оптимизатор (optim.SGD)
        :param epochs: Максимальное число эпох
        :param patience: Число эпох без улучшения перед остановкой
        :param min_delta: Минимальное значимое улучшение
        :param verbose: Выводить ли процесс обучения
        """
        logger.info("Training model with early stopping and regularization...")
        no_improvement = 0

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = self(batch_X)
                loss = criterion(y_pred, batch_y) + self.regularization_loss()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # Логирование прогресса
            if verbose and epoch % 2 == 0:
                logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}")

            # Early stopping логика
            if avg_loss < self.best_loss - min_delta:
                self.best_loss = avg_loss
                no_improvement = 0
                self.best_state_dict = self.state_dict()
            else:
                no_improvement += 1

            if no_improvement >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}, best loss: {self.best_loss:.4f}")
                break

        # Восстанавливаем лучшие веса
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)


class LogisticRegression(nn.Module):
    """
    Логистическая регрессия с поддержкой многоклассовой классификации
    """

    def __init__(self, in_features: int, num_classes: int, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели
        :param x: Входные данные для прямого прохода
        :return: Результаты предсказания модели
        """
        return self.linear(x)

    def train_model(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            class_names: list[str],
            verbose: bool = True
    ) -> dict:
        """
        Обучение модели с обязательной валидацией.
        :param train_loader: DataLoader для обучения
        :param val_loader: DataLoader для валидации
        :param optimizer: оптимизатор
        :param epochs: количество эпох
        :param class_names: список имен классов (для confusion matrix)
        :param verbose: выводить ли прогресс
        :return: словарь с историей обучения
        """
        logger.info("Start training logistic regression...")
        history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': [],
        }

        best_val_loss = float('inf')
        best_preds, best_targets = None, None
        best_metrics = None

        # Обучение по эпохам
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            all_preds, all_targets, all_probs = [], [], []

            # разделение на батчи
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.float()
                y_batch = y_batch.view(-1).long()

                optimizer.zero_grad()
                logits = self(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

            avg_train_loss = total_loss / len(train_loader)
            train_metrics = self._compute_metrics(
                np.array(all_targets), np.array(all_preds), np.array(all_probs)
            )
            train_metrics['loss'] = avg_train_loss
            history['train_loss'].append(avg_train_loss)
            history['train_metrics'].append(train_metrics)

            # Валидация
            self.eval()
            val_loss = 0.0
            val_preds, val_targets, val_probs = [], [], []

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.float()
                    y_val = y_val.view(-1).long()
                    logits = self(X_val)
                    loss = self.criterion(logits, y_val)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(y_val.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_metrics = self._compute_metrics(
                np.array(val_targets), np.array(val_preds), np.array(val_probs)
            )
            val_metrics['loss'] = avg_val_loss
            history['val_loss'].append(avg_val_loss)
            history['val_metrics'].append(val_metrics)

            # Сохраняем лучшие метрики по валидационному loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_preds = val_preds
                best_targets = val_targets
                best_metrics = val_metrics

            if verbose:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val F1: {val_metrics['f1_score']:.4f}"
                )

        # Confusion matrix по лучшей валидации
        self._plot_confusion_matrix(best_targets, best_preds, class_names)

        logger.info("Best Validation Metrics:")
        for key, value in best_metrics.items():
            logger.info(f"{key.capitalize()}: {value:.4f}")

        return history

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray
    ) -> dict:
        """"
        Вычисление метрик
        :param y_true: истинные метки
        :param y_pred: предсказанные метки
        :param y_probs: вероятности
        :return: словарь с метриками
        """
        n = self.num_classes
        eps = 1e-7
        accuracy = (y_true == y_pred).mean()
        p_sum = r_sum = f1_sum = 0.0
        for cls in range(n):
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            p_sum += precision
            r_sum += recall
            f1_sum += f1
        precision = p_sum / n
        recall = r_sum / n
        f1_score = f1_sum / n
        roc_auc = self._multiclass_roc_auc(y_true, y_probs)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc
        }

    def _multiclass_roc_auc(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Вычисление ROC AUC для мультиклассовой задачи
        :param y_true: Истинные метки
        :param y_probs: Предсказанные вероятности
        :return: Значение ROC AUC
        """
        n = self.num_classes
        y_onehot = np.eye(n)[y_true]
        aucs = []
        for i in range(n):
            if np.sum(y_onehot[:, i]) == 0:
                continue
            aucs.append(self._binary_roc_auc(y_onehot[:, i], y_probs[:, i]))
        return float(np.mean(aucs)) if aucs else 0.5

    def _binary_roc_auc(self, y: np.ndarray, scores: np.ndarray) -> float:
        """
        Вычисление ROC AUC для задачи бинарной классификации
        :param y_true: Истинные метки
        :param y_probs: Предсказанные вероятности
        :return: Значение ROC AUC
        """
        order = np.argsort(-scores)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = tp / tp[-1]
        fpr = fp / fp[-1]
        tpr = np.concatenate([np.array([0.0]), tpr, np.array([1.0])])
        fpr = np.concatenate([np.array([0.0]), fpr, np.array([1.0])])
        return float(np.trapezoid(tpr, fpr))

    def _plot_confusion_matrix(
            self,
            y_true: list[int],
            y_pred: list[int],
            class_names: list[str],
            filename: str = "confusion_matrix.png"
    ) -> None:
        """
        Построение матрицы ошибок
        :param y_true: Правильные метки
        :param y_pred: Предсказанные метки
        :param class_names: Количество классов
        :param filename: Имя файла при сохранении
        :return:
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Best Epoch)')
        plt.colorbar()

        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=45)
        plt.yticks(ticks, class_names)

        # Числовые подписи в ячейки
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black')

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':

    # --- Линейная регрессия ---
    # Загрузка данных
    X, y = make_regression_data(source="diabetes")
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Инициализация модели
    model = LinearRegression(
        in_features=X.shape[1],
        regularization='l1',  # Можно выбрать 'l1', или 'l2'
        l1_lambda=0.01,
        l2_lambda=0.01
    )

    # Критерий и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучение модели
    model.train_model(
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=150,
        patience=5,
        verbose=True
    )

    # Сохранение модели
    torch.save(model.state_dict(), 'linreg_torch.pth')

    # Загрузка модели
    loaded_model = LinearRegression(in_features=X.shape[1])
    loaded_model.load_state_dict(torch.load('linreg_torch.pth'))
    loaded_model.eval()


    # # Логистическая регрессия
    iris = IrisDataset(csv_path='datasets/iris.csv', featureEngineering=True)

    X = iris.X.numpy()
    y = iris.y.numpy()

    # Разделение на тренировочную и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = LogisticRegression(
        in_features=X.shape[1],
        num_classes=3,
        criterion=nn.CrossEntropyLoss(),
    )

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    history = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=10,
        class_names=['0', '1', '2'],
        verbose=True
    )

    torch.save(model.state_dict(), 'multiclass_logreg_torch.pth')

    loaded_model = LogisticRegression(in_features=X.shape[1], num_classes=3)
    loaded_model.load_state_dict(torch.load('multiclass_logreg_torch.pth'))
    loaded_model.eval()