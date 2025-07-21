# utility_functions.py
#
# @date : 2024/11
# @auther : Aicocco

# import libraries
import os
import sys
import csv
import json
import yaml
import base64
import shutil
import math
import argparse
import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
from pytz import timezone
import cv2 as cv
import rembg
from PIL import Image, ExifTags
from sklearn.cluster import KMeans
from pprint import pprint
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_LIME = (0,255,0)
COLOR_WHITE = (255,255,255)
COLOR_PURPLE = (255,0,255)
COLOR_CREAM = (140,255,255)


# name : create_file_path
def create_file_path(file_dir, file_name):
    return os.path.join(file_dir, file_name)


# name : make_folder
def make_folder(folder_path):
    if not os.path.isdir(folder_path): os.makedirs(folder_path)


# name : remove_duplicate_elements_in_list
def remove_duplicate_elements_in_list(lst_, is_sort=True):
    lst_eff = list(set(lst_))
    if is_sort: lst_eff.sort()
    return lst_eff


# name : base64_to_cv
def base64_to_cv(image_base64):
    image_bytes = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv.imdecode(np_array, cv.IMREAD_COLOR)
    return image_cv


# name : cv_to_base64
def cv_to_base64(image_cv):
    image_bytes = cv.imencode('.jpg', image_cv)[1].tostring()
    image_base64 = base64.b64encode(image_bytes).decode()
    return image_base64


# name : load_image
def load_image(image_file_dir, dict_input):

    file_name = dict_input["file_name"]
    file_name_0 = dict_input["file_name_without_ext"]

    # load image
    image_file_path = create_file_path(image_file_dir, file_name)
    img_pil = Image.open(image_file_path)
    img = convert_PIL_to_cv2(img_pil) # convert(PIL to CV)

    return img


# name : convert_PIL_to_cv2
def convert_PIL_to_cv2(img_pil):
    
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            exif = img_pil._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    img_pil = img_pil.rotate(180, expand=True)
                elif orientation_value == 6:
                    img_pil = img_pil.rotate(270, expand=True)
                elif orientation_value == 8:
                    img_pil = img_pil.rotate(90, expand=True)
                break

    # PIL画像をNumPy配列に変換
    np_array_img = np.array(img_pil)

    # NumPy配列をOpenCV形式に変換
    img_cv2 = cv.cvtColor(np_array_img, cv.COLOR_RGB2BGR)

    return img_cv2


# name : calculate_contrast
def calculate_contrast(img, upper_threhold_cv = 55.0):
    try:
        # グレースケールに変換
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 輝度値の平均と標準偏差を計算
        mean, stddev = cv.meanStdDev(gray_img)
        sigma = float(stddev[0][0])
        mu = float(mean[0][0])
        coeff_of_variation = float(sigma / mu) * 100 if mu != 0 else 0 # 変動係数(cv)
        is_acceptable_contrast_in_cv = True if coeff_of_variation < upper_threhold_cv else False

        if mu == 0: return None
        if sigma == 0: return None

        dict_res = {
            "mu": round(mu, 2),
            "sigma": round(sigma, 2),
            "coefficient_of_variation": round(coeff_of_variation, 2),
            "upper_threhold_cv": upper_threhold_cv,
            "is_acceptable_contrast_in_cv": is_acceptable_contrast_in_cv
        }
    except:
        return None
        
    return dict_res


# name : enhance_contrast
# brief : 画像のコントラストの調整
# param alpha: コントラストを調整する係数（1.0以上）
# param beta: 明るさを調整するバイアス（0が基本値）
def enhance_contrast(img, alpha=1.5, beta=0):
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


# name : remove_background
# brief : 背景除去(rembg)
# note : rembg.remove()の出力画像は、PIL（Pillow）形式の画像データ。PIL形式からOpenCV形式に変換する必要あり
def remove_background(img):
    img_pil = rembg.remove(img)
    img = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    return img


# name : trim_image
def trim_image(img, x_ratio, y_ratio):
    height, width, _ = img.shape
    x1, y1, x2, y2 = int(x_ratio[0] * width), int(y_ratio[0] * height), int(x_ratio[1] * width), int(y_ratio[1] * height)
    trimmed_img = img[y1:y2, x1:x2]
    
    return trimmed_img


# name : vec_to_df_freq
def vec_to_df_freq(vec):
    try:
        freq = Counter(vec)
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(freq, columns=["Item", "Count"])
    except:
        return None
    return df


# name : safe_cast_to_integer
def safe_cast_to_integer(value, default_value=0):
    if value is None:
        return default_value
    try:
        return int(value)
    except (ValueError, TypeError):
        return default_value  # 変換できない場合はデフォルト値を返す


# name : draw_dashed_line
def draw_dashed_line(img, start_point, end_point, color=(255, 0, 0), thickness=1, gap=5):
    """
    画像に破線を追加する関数

    Parameters:
    img (numpy.ndarray): 画像
    start_point (tuple): 始点 (x, y)
    end_point (tuple): 終点 (x, y)
    color (tuple): 線の色 (B, G, R)
    thickness (int): 線の太さ
    gap (int): 線と線の間隔

    Returns:
    numpy.ndarray: 破線が追加された画像
    """
    
    # 線を引くためのベクトルを計算
    line_vector = np.array(end_point) - np.array(start_point)
    line_length = np.linalg.norm(line_vector)
    
    # 破線を引くために、均等に分けるためのステップ
    step_length = thickness + gap
    num_segments = int(line_length // step_length)

    # 破線を描画
    for i in range(num_segments):
        start = tuple(np.array(start_point) + (line_vector / line_length) * (i * step_length))
        end = tuple(np.array(start_point) + (line_vector / line_length) * ((i * step_length) + thickness))
        cv.line(img, tuple(map(int, start)), tuple(map(int, end)), color, thickness)

    return img