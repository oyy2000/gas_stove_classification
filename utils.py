import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample

from torchvision.utils import make_grid
from tqdm import tqdm
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train GasStove Model')
    
    # 添加命令行参数
    parser.add_argument('--downsample', type=bool, default=False, help='Whether to downsample the dataset')
    parser.add_argument('--weight_loss', type=float, default=1.0, help='Weighting factor for loss function')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification decision')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--cut_ambiguous', type=bool, default=True, help='Whether to remove ambiguous samples from the dataset')
    parser.add_argument('--cut_hide', type=bool, default=True, help='Whether to remove samples that are hidden from view')
    
    args = parser.parse_args()
    return args

# 允许加载截断的图像（可选，根据需要使用）
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GasStoveDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, debug=False):
        """
        自定义数据集类。
        
        Args:
            df (pd.DataFrame): 数据集DataFrame。
            image_dir (str): 图像保存目录路径。
            transform (callable, optional): 图像转换函数。
            debug (bool, optional): 是否启用调试打印。
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        original_index = self.df.iloc[idx]['original_index']
        image_path = os.path.join(self.image_dir, f'image_{original_index}.jpg')
        
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                if self.transform:
                    image = self.transform(image)
        except (OSError, IOError) as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个全黑图像作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)

        # Label 1: 合格/不合格
        label_1 = 1 if self.df.iloc[idx]['是否合格.2'] == '合格' else 0

        # Label 2: 不合格原因
        if label_1 == 1:
            label_2 = 4  # 无不合格原因
        else:
            reason = self.df.iloc[idx]['不合格原因.2']
            if reason == '被挡住无法看出':
                label_2 = 0
            elif reason == '无熄火保护装置':
                label_2 = 1
            elif reason == '猛火灶':
                label_2 = 2
            elif reason == '模糊无法看出':
                label_2 = 3
            else:
                raise ValueError(f"Unrecognized reason '{reason}' at index {idx}")

        # 调试打印
        if self.debug and idx < 5:  # 仅打印前5个样本
            if label_1 == 1:
                print(f"Index {idx}: 合格, label_2 set to 4 (无不合格原因)")
            else:
                print(f"Index {idx}: 不合格, 原因: {reason}, label_2 set to {label_2}")

        return image, torch.tensor(label_1, dtype=torch.float), torch.tensor(label_2, dtype=torch.long)


def create_dataloaders(df, image_save_dir, transform, batch_size=16, num_workers=0):
    """
    划分数据集并创建DataLoader。
    
    Args:
        df (pd.DataFrame): 数据集DataFrame。
        image_save_dir (str): 图像保存目录路径。
        transform (callable): 图像转换函数。
        batch_size (int, optional): 批次大小。
        num_workers (int, optional): DataLoader的工作线程数。
    
    Returns:
        tuple: 训练集、验证集和测试集的DataLoader。
    """
    # 划分数据集
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['是否合格.2']
    )
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_val_df['是否合格.2']
    )

    # 打印数据集大小
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # 打印标签分布
    print("Train Label 1 distribution:")
    print(train_df['是否合格.2'].value_counts())
    print("Validation Label 1 distribution:")
    print(val_df['是否合格.2'].value_counts())
    print("Test Label 1 distribution:")
    print(test_df['是否合格.2'].value_counts())

    # 创建数据集
    train_dataset = GasStoveDataset(train_df, image_save_dir, transform=transform, debug=True)
    val_dataset = GasStoveDataset(val_df, image_save_dir, transform=transform, debug=False)
    test_dataset = GasStoveDataset(test_df, image_save_dir, transform=transform, debug=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def define_model(num_classes_reason=5):
    """
    定义并返回GasStoveClassifier模型。
    
    Args:
        num_classes_reason (int, optional): 不合格原因的类别数。
    
    Returns:
        nn.Module: 定义好的模型。
    """
    class GasStoveClassifier(nn.Module):
        def __init__(self, num_classes_reason=5):
            super(GasStoveClassifier, self).__init__()
            self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.feature_extractor.fc = nn.Identity()  # 移除最后一层
            
            self.classifier_1 = nn.Linear(2048, 1)  # Binary classifier for 合格/不合格
            self.classifier_2 = nn.Linear(2048, num_classes_reason)  # Multi-class classifier for 不合格原因

        def forward(self, x):
            features = self.feature_extractor(x)
            output_1 = torch.sigmoid(self.classifier_1(features))  # Binary output
            output_2 = self.classifier_2(features)  # Multi-class output
            return output_1, output_2
    
    model = GasStoveClassifier(num_classes_reason=num_classes_reason)
    return model


def compute_metrics(predictions, labels, threshold=0.5):
    """
    计算分类指标。
    
    Args:
        predictions (torch.Tensor): 模型的预测输出。
        labels (torch.Tensor): 真实标签。
        threshold (float, optional): 阈值，用于二分类。
    
    Returns:
        tuple: 精确度、召回率、F1分数和准确率。
    """
    preds = (predictions > threshold).float().cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    recall = recall_score(labels, preds, average='binary', zero_division=0)
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    fpr = 1 - recall
    return precision, recall, f1, accuracy, fpr


def create_output_dir(args):
    # 使用当前日期时间戳，保证唯一性
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 将参数信息拼接进文件夹名字中
    folder_name = f"model_downsample_{args.downsample}_wloss_{args.weight_loss} \
    _th_{args.threshold}_cut_ambigious_{args.cut_ambigious}_cut_hide_{args.cut_hide}_{time_str}"
    
    # 创建文件夹
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    return folder_name


def train_model(model, train_loader, val_loader, epochs, device, args=None):
    """
    训练模型，并绘制loss、ROC AUC曲线保存。
    """
        # 定义损失函数和优化器
    criterion_1 = nn.BCELoss()  # 一级分类损失
    if args.weight_loss != 1.0:
        criterion_1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.weight_loss))
    criterion_2 = nn.CrossEntropyLoss(ignore_index=4)  # 二级分类损失，忽略label_2=4的样本
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 创建输出目录（只在训练开始时创建一次）
    output_dir = create_output_dir(args)  
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = 0.0
    best_model_path = None

    # 用于记录训练过程的loss和指标
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "fpr": []}
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, ncols=100)
        
        for batch_idx, (images, label_1, label_2) in enumerate(train_bar):
            images = images.to(device)
            label_1 = label_1.to(device).unsqueeze(1)
            label_2 = label_2.to(device)

            optimizer.zero_grad()
            output_1, output_2 = model(images)
            loss_1 = criterion_1(output_1, label_1)
            loss_2 = criterion_2(output_2, label_2)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算指标
            precision, recall, f1, accuracy,fpr = compute_metrics(output_1, label_1)
            train_metrics["precision"].append(precision)
            train_metrics["recall"].append(recall)
            train_metrics["f1"].append(f1)
            train_metrics["accuracy"].append(accuracy)
            train_metrics["fpr"].append(fpr)
            
            train_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1": f"{f1:.4f}",
                "Acc": f"{accuracy:.4f}",
                "FPR": f"{fpr:.4f}"
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_f1 = sum(train_metrics["f1"]) / len(train_metrics["f1"])
        train_losses.append(avg_train_loss)
        train_f1_scores.append(avg_train_f1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "fpr": []}

        # 为了计算ROC/AUC，需要存储所有预测和标签
        all_val_preds = []
        all_val_labels = []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, ncols=100)

        with torch.no_grad():
            for images, label_1, label_2 in val_bar:
                images = images.to(device)
                label_1 = label_1.to(device).unsqueeze(1)
                label_2 = label_2.to(device)

                output_1, output_2 = model(images)
                loss_1 = criterion_1(output_1, label_1)
                loss_2 = criterion_2(output_2, label_2)
                loss = loss_1 + loss_2

                val_loss += loss.item()

                # 计算指标
                precision, recall, f1, accuracy, fpr = compute_metrics(output_1, label_1)
                val_metrics["precision"].append(precision)
                val_metrics["recall"].append(recall)
                val_metrics["f1"].append(f1)
                val_metrics["accuracy"].append(accuracy)

                val_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "F1": f"{f1:.4f}",
                    "Acc": f"{accuracy:.4f}",
                    "FPR": f"{fpr:.4f}"
                })

                # 收集用于计算ROC的预测和标签
                preds = output_1.detach().cpu().numpy().ravel()  
                labels = label_1.detach().cpu().numpy().ravel()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = sum(val_metrics["f1"]) / len(val_metrics["f1"])
        val_losses.append(avg_val_loss)
        val_f1_scores.append(avg_val_f1)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train F1: {avg_train_f1:.4f}, Val F1: {avg_val_f1:.4f}")

        # 保存最佳模型
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(output_dir, f"best_gas_stove_model_{time_str}.pth")
            torch.save(model.state_dict(), best_model_path)
            print("Saved Best Model!")

    # 所有epoch结束后绘制并保存loss曲线
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_fig_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_fig_path)
    plt.close()

    # 绘制F1曲线
    plt.figure()
    plt.plot(range(1, epochs+1), train_f1_scores, label='Train F1')
    plt.plot(range(1, epochs+1), val_f1_scores, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    f1_fig_path = os.path.join(output_dir, 'f1_curve.png')
    plt.savefig(f1_fig_path)
    plt.close()

    # 返回最佳模型路径
    return best_model_path, output_dir


def evaluate_model(model, test_loader, compute_metrics_fn, device, output_dir=None):
    """
    在测试集上评估模型性能。
    
    Args:
        model (nn.Module): 已训练的模型。
        test_loader (DataLoader): 测试集的DataLoader。
        compute_metrics_fn (callable): 计算指标的函数。
        device (torch.device): 评估设备。
    
    Returns:
        None
    """
    model.eval()
    test_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": []}
    conf_matrix = None

    with torch.no_grad():
        for images, label_1, label_2 in test_loader:
            images = images.to(device)
            label_1 = label_1.to(device).unsqueeze(1)
            label_2 = label_2.to(device)

            output_1, output_2 = model(images)
            precision, recall, f1, accuracy = compute_metrics_fn(output_1, label_1)

            test_metrics["precision"].append(precision)
            test_metrics["recall"].append(recall)
            test_metrics["f1"].append(f1)
            test_metrics["accuracy"].append(accuracy)

            # 计算混淆矩阵
            preds = (output_1 > 0.5).float().cpu().numpy()
            labels = label_1.cpu().numpy()
            if conf_matrix is None:
                conf_matrix = confusion_matrix(labels, preds)
            else:
                conf_matrix += confusion_matrix(labels, preds)

    # 聚合测试指标
    avg_test_metrics = {k: sum(v)/len(v) for k, v in test_metrics.items()}
    print(f"Test Metrics: Precision: {avg_test_metrics['precision']:.4f}, Recall: {avg_test_metrics['recall']:.4f}, F1: {avg_test_metrics['f1']:.4f}, Accuracy: {avg_test_metrics['accuracy']:.4f}")

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(conf_matrix)
    # save the metrics and confusion matrix into a text file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Test Metrics: Precision: {avg_test_metrics['precision']:.4f}, Recall: {avg_test_metrics['recall']:.4f}, F1: {avg_test_metrics['f1']:.4f}, Accuracy: {avg_test_metrics['accuracy']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))


def show_images(dataset, num_images=8):
    """
    显示数据集中的图像网格。
    
    Args:
        dataset (Dataset): 数据集对象（例如，train_dataset, val_dataset）。
        num_images (int, optional): 显示的图像数量。
    
    Returns:
        None
    """
    # 获取一批图像和标签
    images, labels_1, labels_2 = zip(*[dataset[i] for i in range(num_images)])

    # 转换为网格
    images = torch.stack(images)  # 组合张量
    grid = make_grid(images, nrow=4, normalize=True, scale_each=True)

    # 绘制网格
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0))  # 转换为HWC格式用于matplotlib
    plt.axis("off")
    plt.title("Sample Images with Labels")
    plt.show()

    # 显示标签
    for i in range(num_images):
        print(f"Image {i+1}: Label 1 (合格/不合格) = {labels_1[i].item()}, Label 2 (Reason) = {labels_2[i].item()}")
        
def load_and_preprocess_data(data_path, image_save_dir, args):
    """
    加载数据集，验证图像完整性，过滤无效样本，并进行数据平衡。
    
    Args:
        data_path (str): 数据集Excel文件路径。
        image_save_dir (str): 图像保存目录路径。
    
    Returns:
        pd.DataFrame: 经过验证和平衡后的数据集。
    """
    # 加载数据集
    df = pd.read_excel(data_path)
    print(f"Original dataset size: {len(df)}")
    df = df.dropna(subset=['煤气灶图片地址'])  # 确保图像地址不为NaN
    print(f"Dataset size after dropping NaN 煤气灶图片地址: {len(df)}")
    df = df[['煤气灶图片地址', '是否合格.2', '不合格原因.2']]  # 仅保留必要列
    df = df.dropna(subset=['是否合格.2'])  # 确保 '是否合格.2' 不为NaN
    print(f"Dataset size after dropping NaN 是否合格.2: {len(df)}")
    
    # 如果是否合格.2 为 '合格'，则不合格原因.2 必须为 'na'
    # 根据您的数据，调整为实际的NaN或特定字符串
    df.loc[df['是否合格.2'] == '合格', '不合格原因.2'] = 'na'
    print(f"Dataset size after setting '不合格原因.2' to 'na' for '合格' samples: {len(df)}")
    # 当是否合格.2 为 '不合格'， '不合格原因.2' 不能为空
    df = df[~((df['是否合格.2'] == '不合格') & (df['不合格原因.2'].isna()))]
    print(f"Dataset size after dropping NaN 不合格原因.2 for '不合格' samples: {len(df)}")
    # 添加原始索引作为新的列
    df = df.reset_index().rename(columns={'index': 'original_index'})
    
    # 确保有保存图片的目录
    os.makedirs(image_save_dir, exist_ok=True)
    
    # def is_image_valid(image_path):
    #     try:
    #         with Image.open(image_path) as img:
    #             img.load()  # 完全加载图像数据以确保其完整性
    #         return True
    #     except (OSError, IOError) as e:
    #         print(f"Image validation failed for {image_path}: {e}")
    #         return False
    
    # # 过滤出有效的图片，保留原始索引
    # df_valid = df[df['original_index'].apply(
    #     lambda x: is_image_valid(os.path.join(image_save_dir, f'image_{x}.jpg'))
    # )].reset_index(drop=True)
    df_valid = df
    print(f"Valid images after enhanced validation: {len(df_valid)}")
    
    # 打印标签分布
    print("是否合格.2 分布:")
    print(df_valid['是否合格.2'].value_counts())
    print("不合格原因.2 分布 (仅不合格样本):")
    print(df_valid[df_valid['是否合格.2'] == '不合格']['不合格原因.2'].value_counts())
    if args.cut_ambiguous:
        # 去掉不合格原因.2 为 '模糊无法看出' 的样本
        df_valid = df_valid[~((df_valid['是否合格.2'] == '不合格') & (df_valid['不合格原因.2'] == '模糊无法看出'))]
        print(f"Dataset size after dropping '模糊无法看出' samples: {len(df_valid)}")
    
    if args.cut_hide:
        # 去掉不合格原因.2 为 '被挡住无法看出' 的样本
        df_valid = df_valid[~((df_valid['是否合格.2'] == '不合格') & (df_valid['不合格原因.2'] == '被挡住无法看出'))]
        print(f"Dataset size after dropping '被挡住无法看出' samples: {len(df_valid)}")
    # 数据平衡：过采样少数类
    df_majority = df_valid[df_valid['是否合格.2'] == '不合格']
    df_minority = df_valid[df_valid['是否合格.2'] == '合格']
    
    if args.downsample == False:
        df_minority_oversampled = resample(
            df_minority, 
            replace=True,  # 允许重复采样
            n_samples=len(df_majority),  # 使少数类数量与多数类相同
            random_state=42
        )
        df_balanced = pd.concat([df_majority, df_minority_oversampled]).reset_index(drop=True)
    else:
        df_majority_undersampled = resample(
            df_majority, 
            replace=False,  # 不允许重复采样
            n_samples=len(df_minority),  # 使多数类数量与少数类相同
            random_state=42
        )
        df_balanced = pd.concat([df_majority_undersampled, df_minority]).reset_index(drop=True)

    print(f"Balanced dataset size: {len(df_balanced)}")
    print("Balanced Label 1 distribution:")
    print(df_balanced['是否合格.2'].value_counts())
    
    return df_balanced