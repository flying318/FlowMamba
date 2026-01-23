import random
import numpy as np
from PIL import Image
# from scipy import misc
import torch
import torchvision

from PIL import ImageEnhance
import albumentations as A

def affin(img1, img2, angle=0, shear=5, scale=1.0, translate=5):
    """
    对输入的两张图片img1和img2进行随机仿射变换。
    Args:
        img1 (np.ndarray): 第一张图片, 应为np.ndarray类型。
        img2 (np.ndarray): 第二张图片, 应为np.ndarray类型。
        angle (float, optional): 旋转角度范围, 默认为0。
        shear (float, optional): 错切角度范围, 默认为5。
        scale (float, optional): 缩放比例范围, 默认为1.0。
        translate (int, optional): 平移像素范围, 默认为5。
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 返回经过随机仿射变换后的两张图片img1和img2, 均为np.ndarray类型。
    """
    if translate == 0:
        return img1, img2
    dx, dy = random.choice([-translate, translate]), random.choice([-translate, translate])
    # 定义变换
    transform = A.Compose([
        A.Affine(
            scale=(1.0, scale),
            translate_px=(dx, dy),
            rotate=(-angle, angle),
            shear=(-shear, shear),
            p=1
        )
    ])

    # 应用变换
    augmented = transform(image=img2)
    img2 = augmented['image']
    # if random.choice([True, False]):
    #     augmented = transform(image=img1)
    #     img1 = augmented['image']
    # else:
    #     augmented = transform(image=img2)
    #     img2 = augmented['image']

    return img1, img2

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(img.shape[-1]):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

def denormalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(img.shape[0]):  # Loop over color channels
        # normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
        normalized_img[i, ...] = img_array[i, ...] * std[i] + mean[i]
    
    return normalized_img

def save_affin(img, path):
    """Save augmented image."""
    img = denormalize_img(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    img.save(path)
    return


def minmax_normalize_img(img, min_val=0, max_val=255):
    """Normalize image by min-max normalization."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(img.shape[-1]):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - min_val[i]) / (max_val[i] - min_val[i])
    
    return normalized_img

def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label

def random_fliplr_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_1, label_2


def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.fliplr(label_cd)
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_cd, label_1, label_2

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label

def random_flipud_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_1, label_2


def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.flipud(label_cd)
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_cd, label_1, label_2


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_rot_bda(pre_img, post_img, label_1, label_2):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()

    return pre_img, post_img, label_1, label_2


def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    k = random.randrange(3) + 1
    
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    label_cd = np.rot90(label_cd, k).copy()

    return pre_img, post_img, label_cd, label_1, label_2


def random_crop(img, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = img.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pad_image

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_image[H_start:H_end, W_start:W_end, 0]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    img = pad_image[H_start:H_end, W_start:W_end, :]

    return img


def random_bi_image_crop(pre_img, object, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = object.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    H_start = random.randrange(0, H - crop_size + 1, 1)
    H_end = H_start + crop_size
    W_start = random.randrange(0, W - crop_size + 1, 1)
    W_end = W_start + crop_size

    # H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)

    pre_img = pre_img[H_start:H_end, W_start:W_end, :]
    # post_img = post_img[H_start:H_end, W_start:W_end, :]
    object = object[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(GT))
    return pre_img, object


def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
   
    return pre_img, post_img, label


def random_crop_bda(pre_img, post_img, loc_label, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_loc_label = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_clf_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label
    pad_clf_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = clf_label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]
    clf_label = pad_clf_label[H_start:H_end, W_start:W_end]

    return pre_img, post_img, loc_label, clf_label

def random_crop_corn(pre_img, post_img, loc_label, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, pre_img.shape[-1]), dtype=np.float32)

    pad_post_image = np.zeros((H, W, post_img.shape[-1]), dtype=np.float32)
    pad_loc_label = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_clf_label = np.ones((H, W), dtype=np.float32) * ignore_index

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label
    pad_clf_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = clf_label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]
    clf_label = pad_clf_label[H_start:H_end, W_start:W_end]

    return pre_img, post_img, loc_label, clf_label


def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label_1.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label_cd = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_1 = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_label_2 = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img

    pad_label_cd[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_cd
    pad_label_1[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_1
    pad_label_2[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_2

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label_1[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label_cd = pad_label_cd[H_start:H_end, W_start:W_end]
    label_1 = pad_label_1[H_start:H_end, W_start:W_end]
    label_2 = pad_label_2[H_start:H_end, W_start:W_end]

    return pre_img, post_img, label_cd, label_1, label_2

def adjust_brightness(pre_img, post_img, p=0.5):
    if random.random() < p:
        return pre_img, post_img
    pre_img = Image.fromarray(pre_img)
    post_img = Image.fromarray(post_img)
    factor1, factor2 = random.uniform(0.5,2), random.uniform(0.5,2)
    pre_img = ImageEnhance.Brightness(pre_img).enhance(factor1)
    post_img = ImageEnhance.Brightness(post_img).enhance(factor2)
    pre_img = np.asarray(pre_img)
    post_img = np.asarray(post_img)
    return pre_img, post_img

def adjust_contrast(pre_img, post_img, p=0.5):
    if random.random() < p:
        return pre_img, post_img
    pre_img = Image.fromarray(pre_img)
    post_img = Image.fromarray(post_img)
    factor1, factor2 = random.uniform(0.5,2), random.uniform(0.5,2)
    pre_img = ImageEnhance.Contrast(pre_img).enhance(factor1)
    post_img = ImageEnhance.Contrast(post_img).enhance(factor2)
    pre_img = np.asarray(pre_img)
    post_img = np.asarray(post_img)
    return pre_img, post_img

def adjust_sharpness(pre_img, post_img, p=0.5):
    if random.random() < p:
        return pre_img, post_img
    pre_img = Image.fromarray(pre_img)
    post_img = Image.fromarray(post_img)
    factor1, factor2 = random.uniform(0.5,2), random.uniform(0.5,2)
    pre_img = ImageEnhance.Sharpness(pre_img).enhance(factor1)
    post_img = ImageEnhance.Sharpness(post_img).enhance(factor2)
    pre_img = np.asarray(pre_img)
    post_img = np.asarray(post_img)
    return pre_img, post_img

def adjust_color(pre_img, post_img, p=0.5):
    if random.random() < p:
        return pre_img, post_img
    pre_img = Image.fromarray(pre_img)
    post_img = Image.fromarray(post_img)
    factor1, factor2 = random.uniform(0.5,2), random.uniform(0.5,2)
    pre_img = ImageEnhance.Color(pre_img).enhance(factor1)
    post_img = ImageEnhance.Color(post_img).enhance(factor2)
    pre_img = np.asarray(pre_img)
    post_img = np.asarray(post_img)
    return pre_img, post_img