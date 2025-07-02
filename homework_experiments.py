import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from homework_datasets import IrisDataset, ClassificationDataset
from homework_model_modification import LogisticRegression


def run_experiments(
    csv_path: str,
    feature_engineering: bool = False,
    epochs: int = 15
) -> pd.DataFrame:
    """
    Запускает обучение логистической регрессии с различными комбинациями гиперпараметров
    :param csv_path: путь к CSV-файлу
    :param feature_engineering: нужно ли использовать feature engineering
    :param epochs: число эпох
    :return: датафрейм с результатами
    """
    # гиперпараметры
    learning_rates = [0.01, 0.05, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = ['SGD', 'Adam', 'RMSprop']

    iris = IrisDataset(csv_path=csv_path, featureEngineering=feature_engineering)
    X = iris.X.numpy()
    y = iris.y.numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)

    results = []

    # Цикл по всем комбинациям гиперпараметров
    for opt_name in optimizers:
        for lr in learning_rates:
            for bs in batch_sizes:

                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

                model = LogisticRegression(
                    in_features=X.shape[1],
                    num_classes=3
                )
                # Определяю оптимизатор
                if opt_name == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                elif opt_name == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                else:
                    optimizer = optim.RMSprop(model.parameters(), lr=lr)

                # Обучение
                history = model.train_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=epochs,
                    class_names=['0', '1', '2'],
                    verbose=False
                )

                # Беру метрики последней эпохи
                val_metrics = history['val_metrics'][-1]

                results.append({
                    'optimizer': opt_name,
                    'learning_rate': lr,
                    'batch_size': bs,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1_score': val_metrics['f1_score']
                })

    # Возвращаю DataFrame с результатами
    df_results = pd.DataFrame(results)
    return df_results


def plot_results(results_df: pd.DataFrame, feature_engineering: bool) -> None:
    """
    Построение графиков зависимости val_accuracy от batch_size, learning_rate и optimizer
    :param results_df: датафрейм с результатами
    :return: None
    """
    prefix = ""
    if feature_engineering:
        prefix = "feature_engineering_"

    grp_bs = results_df.groupby('batch_size')['val_accuracy'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(grp_bs['batch_size'], grp_bs['val_accuracy'], marker='o')
    plt.title('Val Accuracy vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Val Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{prefix}val_acc_vs_batch_size.png')
    plt.close()

    grp_opt = results_df.groupby('optimizer')['val_accuracy'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(grp_opt['optimizer'], grp_opt['val_accuracy'])
    plt.title('Val Accuracy vs Optimizer')
    plt.xlabel('Optimizer')
    plt.ylabel('Val Accuracy')
    plt.tight_layout()
    plt.savefig(f'{prefix}val_acc_vs_opt.png')
    plt.close()

    grp_lr = results_df.groupby('learning_rate')['val_accuracy'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(grp_lr['learning_rate'], grp_lr['val_accuracy'], marker='o')
    plt.xscale('log')
    plt.title('Val Accuracy vs Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Val Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{prefix}val_acc_vs_lr.png')
    plt.close()

if __name__ == '__main__':
    df = run_experiments(csv_path='datasets/iris.csv', feature_engineering=True, epochs=10)
    plot_results(df, feature_engineering=True)

    print(df)
