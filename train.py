import os
import torch
from torch import nn, optim
from torchvision import transforms
import argparse

from utils import (
    load_and_preprocess_data,
    create_dataloaders,
    define_model,
    compute_metrics,
    train_model,
    evaluate_model,
    parse_args
)


def main():
    args = parse_args()
     # 创建输出文件夹
    
    # 数据路径和图像保存目录
    data_path = 'datasets.xlsx'
    image_save_dir = 'gas_stove_images'

    print("Processing data")
    # 加载和预处理数据 (确保utils.py中已定义好 load_and_preprocess_data 函数)
    df_balanced = load_and_preprocess_data(data_path, image_save_dir, downsample=args.downsample)

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
    # 训练模型
    epochs = args.epochs
    best_model_path, output_dir = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs, 
        device,
        args = args
    )

    # 加载最佳模型（请确保训练过程中保存的模型文件名和这里一致）
    model.load_state_dict(torch.load(best_model_path))

    # 在测试集上评估模型
    evaluate_model(model, test_loader, compute_metrics, device)


if __name__ == "__main__":
    main()


# 被挡住无法看出不作为 negative sample 训练，因为太多样了