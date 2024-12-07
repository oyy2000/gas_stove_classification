import os
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime

from utils import (
    load_and_preprocess_data,
    create_dataloaders,
    define_model,
    compute_metrics,
    evaluate_model,
    parse_args
)

# 允许加载截断的图像（可选，根据需要使用）
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 根据您的环境调整
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            print(f"[WARNING] Error loading image {image_path}: {e}")
            # 返回一个全黑图像作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)

        # Label 1: 合格/不合格
        label_1 = 1 if self.df.iloc[idx]['是否合格.2'] == '合格' else 0

        # Label 2: 不合格原因
        reason = self.df.iloc[idx]['不合格原因.2']
        if label_1 == 1:
            label_2 = 4  # 无不合格原因
        else:
            if pd.isna(reason):
                print(f"[WARNING] '不合格原因.2' is NaN for index {idx}. Assigning label_2 as 4 ('未知').")
                label_2 = 4  # 或者您可以选择一个新的标签
            else:
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
                print(f"[DEBUG] Index {idx}: 合格, label_2 set to 4 (无不合格原因)")
            else:
                print(f"[DEBUG] Index {idx}: 不合格, 原因: {reason}, label_2 set to {label_2}")

        return image, torch.tensor(label_1, dtype=torch.float), torch.tensor(label_2, dtype=torch.long)


def main():
    args = parse_args()
     # 创建输出文件夹
    
    # 数据路径和图像保存目录
    data_path = 'datasets.xlsx'
    image_save_dir = 'gas_stove_images'

    print("Processing data")
    # 加载和预处理数据 (确保utils.py中已定义好 load_and_preprocess_data 函数)
    df_balanced = load_and_preprocess_data(data_path, image_save_dir, args)

    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        df_balanced, 
        image_save_dir, 
        transform, 
        batch_size=16, 
        num_workers=2
    )
    print("DataLoader created")
    
    print("Defining model")
    # 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_model(num_classes_reason=5)
    model = model.to(device)

    print("Begin to train model")

    # 加载最佳模型（请确保训练过程中保存的模型文件名和这里一致）
    model.load_state_dict(torch.load("best_gas_stove_model_20241207_103317.pth"))

    # 在测试集上评估模型
    evaluate_model(model, test_loader, compute_metrics, device, './')
    

if __name__ == "__main__":
    main()
    
    
    
    

# def main():
#     args = parse_args()
#      # 创建输出文件夹
    
#     # 数据路径和图像保存目录
#     data_path = 'datasets.xlsx'
#     image_save_dir = 'gas_stove_images'

#     print("Processing data")
#     # 加载和预处理数据 (确保utils.py中已定义好 load_and_preprocess_data 函数)
#     df_balanced = load_and_preprocess_data(data_path, image_save_dir, downsample=args.downsample)

#     # 图像转换
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # 创建DataLoader
#     train_loader, val_loader, test_loader = create_dataloaders(
#         df_balanced, 
#         image_save_dir, 
#         transform, 
#         batch_size=16, 
#         num_workers=2
#     )
#     print("DataLoader created")
    
#     print("Defining model")
#     # 定义模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = define_model(num_classes_reason=5)
#     model = model.to(device)

#     print("Begin to train model")
#     # 训练模型
#     epochs = args.epochs
#     best_model_path = train_model(
#         model, 
#         train_loader, 
#         val_loader, 
#         epochs, 
#         device,
#         args = args
#     )

#     # 加载最佳模型（请确保训练过程中保存的模型文件名和这里一致）
#     model.load_state_dict(torch.load(best_model_path))

#     # 在测试集上评估模型
#     evaluate_model(model, test_loader, compute_metrics, device)
