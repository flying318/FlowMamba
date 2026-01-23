import os
import numpy as np
import cv2
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import tifffile as tiff
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def class_percentage(in_dir, suffix='.png'):
    # 标签文件夹路径

    # 获取所有标签图像的文件名
    label_files = [f for f in os.listdir(in_dir) if f.endswith(suffix)]

    # 初始化类别计数字典
    class_counts = {}

    # 遍历每个标签图像文件
    for label_file in label_files:
        # 读取标签图像
        label_path = os.path.join(in_dir, label_file)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # 获取图像中所有唯一类别及其计数
        unique, counts = np.unique(label_img, return_counts=True)
        
        # 更新类别计数
        for u, c in zip(unique, counts):
            if u in class_counts:
                class_counts[u] += c
            else:
                class_counts[u] = c

    # 计算每个类别的占比
    total_pixels = sum(class_counts.values())
    class_proportions = {k: v / total_pixels for k, v in class_counts.items()}

    # 输出结果
    print("类别占比:\n",pd.Series(class_proportions))
    print("类别总数:\n", pd.Series(class_counts))
    return

def calculate_std_mean(in_dir, num_bands=3, img_type='png'):
    # 定义转换（转换为Tensor）
    # transform = transforms.Compose([
    #     transforms.ToTensor()  # 转换为 [C, H, W] 的Tensor，像素值范围在[0, 1]
    # ])

    # 初始化累加器
    mean = np.zeros(num_bands)  # 假设是RGB图像
    std = np.zeros(num_bands)
    # mean = torch.zeros(num_bands)  # 假设是RGB图像
    # std = torch.zeros(num_bands)


    # 统计图片总数
    num_images = 0
    def loader(path):
        if img_type !='tif':
            img = cv2.imread(path, -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = tiff.imread(path)
            # img[img<0] = 119.664 
        return img
    # 遍历文件夹中的所有图片
    for filename in tqdm(os.listdir(in_dir)):
        img_path = os.path.join(in_dir, filename)
        
        # 加载图像并转换为Tensor
        image = loader(img_path)
        # image = transform(image)
        
        # 累加图像的均值和标准差
        # mean += image.mean(dim=(1, 2))  # 每个图像的均值
        # std += image.std(dim=(1, 2))    # 每个图像的标准差
        img_mean = image.mean(axis=(0, 1))
        img_std = image.std(axis=(0, 1))
        if np.isnan(img_mean).any() or np.isnan(img_std).any():
            continue
            
        mean += img_mean  # 每个图像的均值
        std += img_std    # 每个图像的标准差
        # img_std[img_std == 0] = 1e-8
        num_images += 1

    # 计算总体均值和标准差
    mean /= num_images
    std /= num_images

    print(f"总体均值: {np.round(mean,3)}")
    print(f"总体标准差: {np.round(std,3)}")
    return

def load_images_from_folder(folder):
    images1 = []
    images2 = []
    for filename in tqdm(os.listdir(os.path.join(folder, 'img1'))):
        img1_path = os.path.join(folder, 'img1', filename)
        img2_path = os.path.join(folder, 'img2', filename)
        img1 = np.array(Image.open(img1_path).convert('L'))
        img2 = np.array(Image.open(img2_path).convert('L'))
        images1.append(img1)
        images2.append(img2)
    return images1, images2

def extract_mean_feature(images):
    mean_features = []
    for img in images:
        mean = np.mean(img)
        mean_features.append(mean)
    return np.array(mean_features)

def plot_kde(train_path, test_path, output_file):
    train_img1, train_img2 = load_images_from_folder(train_path)
    test_img1, test_img2 = load_images_from_folder(test_path)

    train_img1_mean_features = extract_mean_feature(train_img1)
    train_img2_mean_features = extract_mean_feature(train_img2)
    train_mean_features = np.mean([train_img1_mean_features, train_img2_mean_features], axis=0)
    test_img1_mean_features = extract_mean_feature(test_img1)
    test_img2_mean_features = extract_mean_feature(test_img2)
    test_mean_features = np.mean([test_img1_mean_features, test_img2_mean_features], axis=0)

    sns.kdeplot(train_mean_features, fill=True, color='orange', label='train')
    sns.kdeplot(test_mean_features, fill=True, color='purple', label='test')
    plt.xlabel('Pixel Mean')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation of Datasets')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    # in_dir = 'dataset/corn/label'
    in_dir = 'dataset/dfc2025_trainval/train/target'
    # calculate_std_mean(in_dir, 3, 'png')
    class_percentage(in_dir, '.tif')
    # train_dir = 'dataset/jilin/train'
    # test_dir = 'dataset/jilin/test'
    # output_file = 'kde_plot.png'
    # plot_kde(train_dir, test_dir, output_file)