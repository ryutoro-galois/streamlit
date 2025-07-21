# base_functions.py
#
# @date : 2024/11
# @auther : Aicocco
#
#===== Module List =====
# === [Main Modules] ===
# - judge_sheet_condition() : シート状態判定のメインモジュール
#    - edge_folding_check() : 内折れ/外折れ判定モジュール
#    - gap_size_check(): ギャップ判定モジュール
#
# === [Sub Modules] ===
# - draw_edge_from_judge_result(): Edge折れ検知結果の印字
# - remove_background(): 背景除去
# - get_angle(): 2点間の直線の角度
# - fillConvexPoly(): 領域色塗り
# - get_centroids_kmeans(): k-meansクラスタリングによる中心点の計算
# - get_corners_Harris(): Harrisのコーナー検知
# - estimate_sheet_count_trial(): シート枚数の推定(Trial)
# - estimate_sheet_count(): シート枚数の推定
# - get_coordinate_sheet_pair(): 段ボールのシートペア座標の取得
# - get_corners_Harris_sub_region(): サブ領域に対するHarrisのコーナー検知
# - calc_edge_difference(): Edge差分の計算
# - judge_edge_status(): Edge差分値に基づいた状態判定
# - ...
#=======================


# import libraries
from utility_functions import *


# read config
#filedir = os.path.dirname(__file__)
filedir = "/mount/src/streamlit/Documents/streamlit_judge_sheet_condition_app"
filename = "config_sheet_condition.yml"
filepath = f"{filedir}/{filename}"
with open(filepath, encoding='utf-8') as file:
    config = yaml.safe_load(file)

# IS_DEBUG_PRINT
IS_DEBUG_PRINT = config["IS_DEBUG_PRINT"]

# TEST_NAME_OF_SHEET_CONDITION (rename)
TEST_01_EDGE_FOLDING_CHECK = config["TEST_NAME_OF_SHEET_CONDITION"]["EN"]["EDGE_FOLDING_CHECK"]
TEST_02_JOINT_GAP_SIZE_CHECK = config["TEST_NAME_OF_SHEET_CONDITION"]["EN"]["JOINT_GAP_SIZE_CHECK"]

# EDGE_LABEL (rename)
EDGE_LABEL_00_NORMAL = config["EDGE_LABEL"]["JA"]["00_NORMAL"]
EDGE_LABEL_01_INWARD_FOLD = config["EDGE_LABEL"]["JA"]["01_INWARD_FOLD"]
EDGE_LABEL_02_OUTWARD_FOLD = config["EDGE_LABEL"]["JA"]["02_OUTWARD_FOLD"]

# GAP_LABEL (rename)
GAP_LABEL_00_NORMAL = config["GAP_LABEL"]["JA"]["00_NORMAL"]
GAP_LABEL_01_NARROW = config["GAP_LABEL"]["JA"]["01_NARROW"]
GAP_LABEL_02_WIDE = config["GAP_LABEL"]["JA"]["02_WIDE"]


# name : adjust_image_contrast
# brief : コントラスト調整
def adjust_image_contrast(img):
    
    # upper_threhold_cv
    upper_threhold_cv = float(config["IMAGE_CONTRAST_ADJUSTMENT"]["UPPER_THRESHOLD_CV_IN_CONTRAST_ADJUSTMENT"])

    # init
    img0 = img.copy()

    try:
        # initial check
        dict_res = calculate_contrast(img, upper_threhold_cv)
        if IS_DEBUG_PRINT: 
            print("\n=== [START] contrast (brightness) check ===")
            pprint(dict_res)
            print("\n")
        
        if dict_res["is_acceptable_contrast_in_cv"] == False:
            lst_alpha = [round(1.5 + 0.5 * i, 1) for i in range(6)] # 1.5- by 0.5
            for alpha_ in lst_alpha:
                # copy
                img = img0.copy()
                # enhance_contrast
                img = enhance_contrast(img, alpha_)
                # calculate_contrast
                dict_res = calculate_contrast(img, upper_threhold_cv)
                print("alpha = [", alpha_, "]")
                pprint(dict_res)
                if dict_res is None: return img0
                if dict_res["is_acceptable_contrast_in_cv"] == True:
                    print("is_acceptable_contrast_in_cv = [", dict_res["is_acceptable_contrast_in_cv"], "]")
                    break
    except:
        return img0
    
    return img


# name : preprocess
def preprocess(img):

    # adjust_image_contrast
    img = adjust_image_contrast(img)

    # remove_background (1st)
    img = remove_background(img)

    # crop_cardboard_area
    if True:
        img, dict_res = crop_cardboard_area(img)
        if IS_DEBUG_PRINT: 
            print("=== [START] crop_cardboard_area ===")
            pprint(dict_res)
            print("\n")

    return img


# name : draw_edge_from_judge_result
# brief : Edge折れ検知結果の印字
def draw_edge_from_judge_result(img_, df):
    
    # dict_edge_label_color
    dict_edge_label_color = {
        EDGE_LABEL_00_NORMAL: COLOR_WHITE,
        EDGE_LABEL_01_INWARD_FOLD: COLOR_RED,
        EDGE_LABEL_02_OUTWARD_FOLD: COLOR_BLUE,
    }
    
    img = img_.copy()

    dot_size = 20
    for idx, item in df.iterrows():

        judged_edge_status = item["edge"]
        target_color = dict_edge_label_color[judged_edge_status]

        upper_x = int(item["edge_coord_upper_x"])
        lower_x = int(item["edge_coord_lower_x"])

        upper_y = int(0.5 * (int(item["c11_upper_y_in_upper_sheet_pair"]) + int(item["c12_lower_y_in_upper_sheet_pair"])))
        lower_y = int(0.5 * (int(item["c21_upper_y_in_lower_sheet_pair"]) + int(item["c22_lower_y_in_lower_sheet_pair"])))
        
        pt_upper = (upper_x, upper_y)
        pt_lower = (lower_x, lower_y)
    
        img = cv.circle(img, pt_upper, dot_size, target_color, -1)
        img = cv.circle(img, pt_lower, dot_size, target_color, -1)
    
    return img


# name : draw_gap_size_from_judge_result
# brief : Gap Size検知結果の印字
def draw_gap_size_from_judge_result(img_, df):
    
    # dict_gap_label_color
    dict_gap_label_color = {
        GAP_LABEL_00_NORMAL: COLOR_WHITE,
        GAP_LABEL_01_NARROW: COLOR_BLUE,
        GAP_LABEL_02_WIDE: COLOR_RED
    }

    img = img_.copy()

    line_width = 30
    for idx, item in df.iterrows():

        judged_gap_status = item["judge_result_ja"]
        target_color = dict_gap_label_color[judged_gap_status]
    
        gap_lower = safe_cast_to_integer(item["estimated_gap_pixcel"])
        if gap_lower <= 0: continue
        lower_left_x = safe_cast_to_integer(item["estimated_lower_left_x"])
        lower_right_x = safe_cast_to_integer(item["estimated_lower_right_x"])
        lower_y = int(0.5 * (int(item["c21_upper_y_in_lower_sheet_pair"]) + int(item["c22_lower_y_in_lower_sheet_pair"])))
        
        s_pt = (lower_left_x, lower_y)
        e_pt = (lower_right_x, lower_y)

        img = cv.line(img, s_pt, e_pt, target_color, line_width)
    
    # draw gap center line
    if True:
        estimated_gap_center_x = safe_cast_to_integer(df.loc[0,"estimated_gap_center_x"])
        y1 = int(df["c21_upper_y_in_lower_sheet_pair"].min() * 0.95)
        y2 = int(df["c22_lower_y_in_lower_sheet_pair"].max() * 1.05)
        s_pt = (estimated_gap_center_x, y1)
        e_pt = (estimated_gap_center_x, y2)
        target_color = COLOR_WHITE
        target_line_width = 10
        img = draw_dashed_line(img, s_pt, e_pt, target_color, target_line_width, gap=30)

    return img    


# name : draw_sheet_separated_line
def draw_sheet_separated_line(img_, lst_sheet_count):

    # get param
    Flute = lst_sheet_count[0]["Flute"]
    mm = len(Flute)
    mm_times_2 = (2 * mm)

    img = img_.copy()

    cnt = -1 # init (to -1)
    for lst_x in lst_sheet_count:
        y = lst_x["upper_y"]
        x1 = lst_x["min_x"]
        x2 = lst_x["max_x"]
        s_pt = (x1, y)
        e_pt = (x2, y)
        if lst_x["enable"] == False:
            target_color = COLOR_WHITE
            target_line_width = 10
            img = draw_dashed_line(img, s_pt, e_pt, target_color, target_line_width, gap=20)
        else:
            cnt += 1
            target_color = COLOR_RED if cnt % mm_times_2 == mm else COLOR_LIME
            target_line_width = 10 if cnt % mm_times_2 == mm else 2
            if cnt % mm == 0:
                img = cv.line(img, s_pt, e_pt, target_color, target_line_width)
            
    return img


# name : get_angle
# brief : 2点間の直線の角度
# return : [-180.0, 180.0]
def get_angle(pt_1, pt_2, is_abs = False):
    try:
        (x1, y1) = pt_1
        (x2, y2) = pt_2
        # calculate angle from y/x ratio using atan2
        angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        angle = round(angle, 4)
        if is_abs: angle = abs(angle)
    except Exception:
        return None
    return angle


# name : get_centroids_kmeans
# brief : k-meansクラスタリングによる中心点の計算
def get_centroids_kmeans(data, num_clusters, seed=1234):
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(data.reshape(-1, 1))
    centroids = kmeans.cluster_centers_
    return centroids.flatten()


# name : fast_line_detector
def fast_line_detector(img):
    
    length_threshold = 10 # default 10
    distance_threshold = 1.41421356
    canny_th1 = 50.0
    canny_th2 = 50.0
    canny_aperture_size = 3
    do_merge = False
    
    # グレースケールに変換
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Fast Line Detectorを作成
    line_detector = cv.ximgproc.createFastLineDetector(
        length_threshold, distance_threshold, 
        canny_th1, canny_th2, canny_aperture_size, do_merge
    )
    # 直線を検出
    lines = line_detector.detect(gray)

    return lines


# name : crop_cardboard_area
def crop_cardboard_area(img, n_bins=100):

    height, width, _ = img.shape

    # binの幅
    bin_height = height // n_bins
    
    # init
    cols = ["count_all", "count_1", "count_2", "count_3"]
    cols_all = ['segment_id', 'start_y', 'end_y', 'share'] + cols
    df = pd.DataFrame(index=[i for i in range(n_bins)], columns=cols_all)
    for col in cols: 
        df[col] = 0
        df[col+"_flag"] = 0

    # FastLineDetector within [x_delta_minus ~ x_delta_plus]
    lines = fast_line_detector(img)  

    L2_LOWER = 10
    for i in range(n_bins):
        start_y = int(i * bin_height)
        end_y = int((i + 1) * bin_height) if (i + 1) * bin_height <= height else height
        
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            ref_y = int(0.5 * (y1 + y2))
            L2 = int((x2-x1)**2 + (y2-y1)**2)
            if L2 > L2_LOWER and ref_y > start_y and ref_y < end_y:
                degree_abs = get_angle((x1, y1), (x2, y2), is_abs = True)
                if degree_abs is not None:
                    # whether it is the center (wave part) of the cardboard
                    degree_abs = get_angle((x1, y1), (x2, y2), is_abs = True)
                    if (degree_abs < 10 or abs(180 - degree_abs) < 10):
                        df.loc[i,"count_all"] += 1
                        df.loc[i,"count_1"] += 1
                    if (degree_abs > 30 and degree_abs < 60):
                        df.loc[i,"count_all"] += 1
                        df.loc[i,"count_2"] += 1
                    if (degree_abs > 120 and degree_abs < 150):
                        df.loc[i,"count_all"] += 1
                        df.loc[i,"count_3"] += 1

        # set
        df.loc[i,"segment_id"] = i+1
        df.loc[i,"share"] = round((i+1) / n_bins, 2)
        df.loc[i,"start_y"] = start_y
        df.loc[i,"end_y"] = end_y

    fillFlg = df["count_all"] > 0
    df = df[fillFlg].reset_index(drop=True)
    mu_minus_sigma = df[cols].median() - df[cols].std()
    for col in cols:
        varname = col+"_flag"
        fillFlg_l = df[col] > mu_minus_sigma[col]
        if sum(fillFlg_l) > 0: df.loc[fillFlg_l, varname] = 1
    
    # init 
    start_y_upper = end_y_upper = diff_y_upper = 0
    start_y_lower = end_y_lower = diff_y_lower = 0
    for idx, item in df.iterrows():
        if((item["count_all_flag"] == 1) & (item["count_1_flag"] == 1) & (item["count_2_flag"] == 1) & (item["count_3_flag"] == 1)):
            start_y_upper = item["start_y"]
            end_y_upper = item["end_y"]
            diff_y_upper = (end_y_upper - start_y_upper)
            break

    for idx, item in df.iloc[::-1].iterrows():
        if((item["count_all_flag"] == 1) & (item["count_1_flag"] == 1) & (item["count_2_flag"] == 1) & (item["count_3_flag"] == 1)):
            start_y_lower = item["start_y"]
            end_y_lower = item["end_y"]
            diff_y_lower = (end_y_lower - start_y_lower)
            break

    upper_y_0 = start_y_upper - 2 * diff_y_upper
    lower_y_0 = end_y_lower + diff_y_lower
    y_range = [upper_y_0, lower_y_0]
    
    # 元画像をcropした後、cropped画像を元の画像と同じサイズの透明な画像に合成する
    if True:
        y1 = upper_y_0
        y2 = lower_y_0
        cropped_img = img[y1:y2, :]

        # 元の画像と同じサイズの透明な画像を作成 (RGBA)
        transparent_img = np.zeros((height, width, 4), dtype=np.uint8)

        # クロップした画像をアルファチャンネルを持つ形式に変換
        # (height, width, 3)から(height, width, 4)へ変換
        cropped_rgba = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGBA)
        # クロップした画像の位置を決定 (元画像の同じ位置に合成)
        transparent_img[y1:y2, :, :3] = cropped_rgba[:, :, :3]  # RGB部分を合成
        transparent_img[y1:y2, :, 3] = 255  # 最新に設定された部分を不透明にする

    diff_upper_and_lower_y_0 = (lower_y_0 - upper_y_0)
    diff_upper_and_lower_y_0_ratio = round(diff_upper_and_lower_y_0 / height, 3)
    upper_y_0_ratio = round(upper_y_0 / height, 3)
    lower_y_0_ratio = round(lower_y_0 / height, 3)

    # set
    dict_res = {
        "width": width, 
        "height": height, 
        "upper_y_0": upper_y_0,
        "lower_y_0": lower_y_0,
        "diff_upper_and_lower_y_0": diff_upper_and_lower_y_0,
        "upper_y_0_ratio": upper_y_0_ratio,
        "lower_y_0_ratio": lower_y_0_ratio,
        "diff_upper_and_lower_y_0_ratio": diff_upper_and_lower_y_0_ratio
    }

    # remove_background
    transparent_img = remove_background(transparent_img)

    return transparent_img, dict_res


# name : fillConvexPoly
# brief : 領域色塗り
def fillConvexPoly(img, lower_area_ratio=0.01, fill_color=(255,0,0)):
  
    h, w, _ = img.shape
    img_size = h * w

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

    # Opencv ver.4.x
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for l, contour in enumerate(contours):
        m = len(contour)
        if m >= 4:
            x,y,w,h = cv.boundingRect(contour)
            area = w*h
            area_ratio = round(area / img_size, 4)
            if area_ratio < lower_area_ratio:
                img = cv.fillConvexPoly(img, points=contour, color=fill_color)

    if False:
        # 画像をHSV色空間に変換する
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # 白色の範囲を設定する（HSVでの範囲指定）
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        # 白色の領域を抽出する
        mask = cv.inRange(hsv, lower_white, upper_white)
        # 白色の領域を指定した色に変換する
        img[mask != 0] = fill_color

        # 灰色の範囲を広げるための調整値
        lower_adjust = np.array([-10, -10, -10])
        upper_adjust = np.array([10, 10, 10])
        # 灰色の範囲を設定する
        lower_gray = np.array([0, 0, 160]) + lower_adjust
        upper_gray = np.array([180, 20, 200]) + upper_adjust
        # 灰色の領域を抽出する
        mask = cv.inRange(hsv, lower_gray, upper_gray)
        # 白色の領域を指定した色に変換する
        img[mask != 0] = fill_color

    return img


# name : get_corners_Harris_0
def get_corners_Harris_0(img, threshold = 0.01):

    # コーナー検出パラメータを設定
    block_size = 2  # コーナー検出を行うために考慮される近傍領域のサイズ
    ksize = 3  # Sobelオペレータのアパーチャーサイズ
    k = 0.04  # Harris検出器に使用されるfree parameter

    # グレー変換
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # init
    corners = None
    if config["NOISE_REMOVAL_FILTER_TYPE"] == "NON_FILTER":
        # フィルタ処理なし
        corner_img = cv.cornerHarris(gray_img, block_size, ksize, k)
        corners = np.argwhere(corner_img > threshold * corner_img.max())

    elif config["NOISE_REMOVAL_FILTER_TYPE"] == "GAUSSIAN_FILTER":
        # ガウシアンフィルタを適用してノイズを除去
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)
        corner_img = cv.cornerHarris(gray_img, block_size, ksize, k)
        corners = np.argwhere(corner_img > threshold * corner_img.max())  

    elif config["NOISE_REMOVAL_FILTER_TYPE"] == "MEDIAN_FILTER":
        # メディアンフィルタを適用してノイズを除去
        gray_img = cv.medianBlur(gray_img, ksize=3)
        corner_img = cv.cornerHarris(gray_img, block_size, ksize, k)
        corners = np.argwhere(corner_img > threshold * corner_img.max())      

    elif config["NOISE_REMOVAL_FILTER_TYPE"] == "SOBEL_FILTER":
        # Sobelフィルタを適用してノイズを除去
        sobel_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=5)  # X方向
        sobel_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=5)  # Y方向
        sobel_magnitude = cv.magnitude(sobel_x, sobel_y) # Sobelフィルターの結果を合成して強度画像を生成
        corner_img = cv.cornerHarris(sobel_magnitude.astype(np.float32), block_size, ksize, k)
        corners = np.argwhere(corner_img > threshold * corner_img.max())
    
    return corners


# name : get_corners_Harris
# brief : Harrisのコーナー検知
def get_corners_Harris(img, threshold = 0.01):

    # get_corners_Harris_0
    corners = get_corners_Harris_0(img, threshold)
    
    min_x = int(np.min(corners[:,1]))
    max_x = int(np.max(corners[:,1]))
    min_y = int(np.min(corners[:,0]))
    max_y = int(np.max(corners[:,0]))
    
    dict_res = {} # init

    # left_upper (min, min)
    filFlg_upper = (corners[:,0] >= min_y) & (corners[:,0] <= int(1.1*min_y))
    corners_upper = corners[filFlg_upper]
    upper_min_x = int(np.min(corners_upper[:,1]))
    dict_res["left_upper"] = [upper_min_x, min_y]

    # left_lower (min, max)
    filFlg_upper = (corners[:,0] >= int(max_y*0.9)) & (corners[:,0] <= max_y)
    corners_upper = corners[filFlg_upper]
    upper_min_x = int(np.min(corners_upper[:,1]))
    dict_res["left_lower"] = [upper_min_x, max_y]

    # right_upper (max, min)
    filFlg_upper = (corners[:,1] >= int(max_x*0.9)) & (corners[:,1] <= max_x)
    corners_upper = corners[filFlg_upper]
    upper_min_y = int(np.min(corners_upper[:,0]))
    upper_max_y = int(np.max(corners_upper[:,0]))
    dict_res["right_upper"] = [max_x, upper_min_y]  # right_upper (max, min)
    dict_res["right_lower"] = [max_x, upper_max_y]  # right_lower (max, max)

    return dict_res


# name : estimate_sheet_count_trial
# brief: シート枚数の推定(Trial)
def estimate_sheet_count_trial(img, dict_corners, dict_input, num_of_sheets_trial, x_ratio = 0.5):
    
    # get param
    judge_parity = dict_input["judge_parity"]
    Flute = dict_input["Flute"]
    BoxesPerBD = dict_input["BoxesPerBD"]
    
    # x_axis
    min_x = min(int(dict_corners["left_lower"][0]), int(dict_corners["left_upper"][0]))
    max_x = max(int(dict_corners["right_lower"][0]), int(dict_corners["right_upper"][0]))
    min_y = min(int(dict_corners["left_upper"][1]), int(dict_corners["right_upper"][1]))
    max_y = max(int(dict_corners["left_lower"][1]), int(dict_corners["right_lower"][1]))
    x_width = (max_x - min_x)

    # y_axis
    diff_y = max_y - min_y
    diff_y_per_unit = round(diff_y / (2 * BoxesPerBD), 2) # 2*nn
    sheet_thickness_pixcel = diff_y_per_unit  # 画像検知結果から取得(pixcel単位)
     
    if judge_parity == "left":
        x_delta_plus = int(min_x + x_ratio * x_width)
        x_delta_minus = max(min_x, int(min_x - x_ratio * x_width))
    elif judge_parity == "right":
        x_delta_plus = max(max_x, int(max_x + x_ratio * x_width))
        x_delta_minus = int(max_x - x_ratio * x_width)
        
    # FastLineDetector within [x_delta_minus ~ x_delta_plus]
    lines = fast_line_detector(img)

    L2_LOWER = 50
    lst_y = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            x_avg = int((x1+x2)*0.5)
            y_avg = int((y1+y2)*0.5)
            L2 = int((x2-x1)**2 + (y2-y1)**2)
            if x_avg > x_delta_minus and x_avg < x_delta_plus and L2 > L2_LOWER:
                degree_abs = get_angle((x1, y1), (x2, y2), is_abs = True)
                if degree_abs is not None and (degree_abs < 10 or abs(180 - degree_abs) < 10):
                    lst_y.append(y1)
                    lst_y.append(y2)

    # remove_duplicate_elements_in_list
    lst_y_eff = remove_duplicate_elements_in_list(lst_y)
    
    lst_ = lst_y_eff
    num_clusters = int(num_of_sheets_trial + 1) # (num_of_sheets + 1)
    centroids = get_centroids_kmeans(np.array(lst_), num_clusters, seed=1234) # get_centroids_kmeans
    lst_centroids = list(centroids)
    lst_centroids.sort()
    lst_centroids = list(map(lambda x: int(x), lst_centroids)) # float to int
    
    lst_dict_count_by_sheet = [] #init
    lst_count = []
    nn = len(lst_centroids)
    for idx in range(nn-1):
        ref_y1 = lst_centroids[idx]
        ref_y2 = lst_centroids[idx+1]
        delta_y21 = (ref_y2 - ref_y1)
        dict_count_by_sheet = {} # init
        tmp_count = 0
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            L2 = int((x2-x1)**2 + (y2-y1)**2)
            if y1 > ref_y1 and y1 < ref_y2 and L2 > 10:
                degree_abs = get_angle((x1, y1), (x2, y2), is_abs = True)
                if degree_abs is not None:
                    # whether it is the center (wave part) of the cardboard
                    if ((degree_abs > 30 and degree_abs < 60) or (degree_abs > 120 and degree_abs < 150)):
                        tmp_count += 1
        # set
        dict_count_by_sheet = {
            "Flute": Flute,
            "BoxesPerBD": BoxesPerBD,
            "judge_parity": judge_parity, 
            "num_of_sheets": num_of_sheets_trial,
            "sheet_thickness_pixcel": sheet_thickness_pixcel,
            "id": int(idx+1),
            "count": tmp_count,
            "enable": True, # init
            "upper_y": ref_y1,
            "lower_y": ref_y2,
            "delta_y": delta_y21, 
            "height_0": diff_y, 
            "min_x": min_x,
            "max_x": max_x
        }
        lst_count.append(tmp_count)
        lst_dict_count_by_sheet.append(dict_count_by_sheet)
        
        # Whether it is the center (wave part) of the cardboard
        #  - enable = True: cardboard
        #  - enable = False: Hollow part that is not cardboard
        count_med = np.median(lst_count)
        count_std = round(np.std(lst_count),1)
        ##count_2sigma = count_med - 2 * count_std
        count_2sigma = count_med - 1.5 * count_std

        id_eff = 0
        for lst_x in lst_dict_count_by_sheet:
            count_x = lst_x["count"]
            lst_x["id_eff"] = 0
            if count_x < count_2sigma:
                lst_x["enable"] = False
            else:
                id_eff += 1
                lst_x["id_eff"] = id_eff

        cnt_eff = 0
        lst_dict_count_by_sheet_eff = [] # init
        for lst_x in lst_dict_count_by_sheet:
            if lst_x["enable"]:
                cnt_eff += 1
                lst_dict_count_by_sheet_eff.append(lst_x) 

    return lst_dict_count_by_sheet, lst_dict_count_by_sheet_eff


# name : estimate_sheet_count_equally
# brief : 均等割
def estimate_sheet_count_equally(img, dict_corners, dict_input):

    # get param
    judge_parity = dict_input["judge_parity"]
    Flute = dict_input["Flute"]
    BoxesPerBD = dict_input["BoxesPerBD"]
    num_of_sheets = 2 * BoxesPerBD

    # x_axis
    min_x = min(int(dict_corners["left_lower"][0]), int(dict_corners["left_upper"][0]))
    max_x = max(int(dict_corners["right_lower"][0]), int(dict_corners["right_upper"][0]))
    min_y = min(int(dict_corners["left_upper"][1]), int(dict_corners["right_upper"][1]))
    max_y = max(int(dict_corners["left_lower"][1]), int(dict_corners["right_lower"][1]))
    x_width = (max_x - min_x)

    # y_axis
    diff_y = max_y - min_y
    diff_y_per_unit = round(diff_y / (2 * BoxesPerBD), 2) # 2*nn
    sheet_thickness_pixcel = diff_y_per_unit  # 画像検知結果から取得(pixcel単位)

    lst_dict_count_by_sheet = [] #init
    for l in range(num_of_sheets):
        ref_y1 = int(max_y - diff_y_per_unit * ((num_of_sheets)-l)) # upper line
        ref_y2 = int(max_y - diff_y_per_unit * ((num_of_sheets)-(l+1))) # lower line
        delta_y21 = (ref_y2 - ref_y1)
        # set
        dict_count_by_sheet = {
            "Flute": Flute,
            "BoxesPerBD": BoxesPerBD,
            "judge_parity": judge_parity, 
            "num_of_sheets": num_of_sheets,
            "sheet_thickness_pixcel": sheet_thickness_pixcel,
            "id": int(l+1),
            "count": 0,
            "enable": True, # init
            "upper_y": ref_y1,
            "lower_y": ref_y2,
            "delta_y": delta_y21, 
            "height_0": diff_y, 
            "min_x": min_x,
            "max_x": max_x
        }
        lst_dict_count_by_sheet.append(dict_count_by_sheet)

    return lst_dict_count_by_sheet


# name : estimate_sheet_count
# brief: シート枚数の推定
def estimate_sheet_count(img, dict_input, threshold_corners_Harris = 0.001):

    # init
    dict_input["sheet_count_check"] = False

    # get param
    judge_parity = dict_input["judge_parity"]
    Flute = dict_input["Flute"]
    BoxesPerBD = dict_input["BoxesPerBD"]
    nn = BoxesPerBD

    # Flute in ('A','B','C') -> 2*nn
    # Flute in ('AB','CB',..) -> 4*nn
    start_nn = (2 * len(Flute) * nn)
    #start_nn = (3 * len(Flute) * nn)
    #end_nn = start_nn + nn
    end_nn = start_nn + int(0.7 * nn)
    #end_nn = start_nn + int(0.5 * nn)

    # get_corners_Harris
    dict_corners = get_corners_Harris(img, threshold_corners_Harris)

    if IS_DEBUG_PRINT: 
        print("=== [START] estimate_sheet_count ===")

    target_num_of_sheets = start_nn
    for i in range(end_nn, start_nn-1, -1): # subtraction loop
        num_of_sheets_trial = i
        lst_sheet_count, lst_sheet_count_eff = estimate_sheet_count_trial(img, dict_corners, dict_input, num_of_sheets_trial)
        sheet_count_eff = len(lst_sheet_count_eff)
        
        # output png
        if False:
        #if config["IS_DEBUG_PRINT"]:
            # draw_sheet_separated_line
            img_01 = draw_sheet_separated_line(img, lst_sheet_count)
            # output png
            file_name_0 = dict_input["file_name_without_ext"]
            output_dir = "./Pyin"
            output_name = f"img_sheet_count_{file_name_0}_num_sheet_{num_of_sheets_trial}.png"
            output_path = create_file_path(output_dir, output_name)
            cv.imwrite(output_path, img_01)
            print(output_name)

        
        if IS_DEBUG_PRINT:
            print("num_of_sheets_trial = [", num_of_sheets_trial, "]")
            print("sheet_count_eff = [", sheet_count_eff, "]")
            print("diff = [", (sheet_count_eff - target_num_of_sheets), "]")
        if sheet_count_eff == target_num_of_sheets: 
            dict_input["sheet_count_check"] = True
            pprint(lst_sheet_count_eff)
            #sys.exit()
            if False:
                print("sheet_count_check = [", dict_input["sheet_count_check"], "]")
                ##print("sheet_count_eff = [", sheet_count_eff, "]")
            break
    
    
    # 隙間部分を除いてシート当たりの厚み(pixcel)を再算出
    if dict_input["sheet_count_check"]:
        sheet_thickness_pixcel_total = 0
        for lst_x in lst_sheet_count:
            if lst_x["enable"]: 
                sheet_thickness_pixcel_total += lst_x["delta_y"]
        diff_y = lst_sheet_count[0]["height_0"]
        if diff_y > sheet_thickness_pixcel_total:
            sheet_thickness_pixcel_rev = round(sheet_thickness_pixcel_total / (2 * BoxesPerBD), 2) # 2*nn
            if False:
                sheet_thickness_pixcel_0 = lst_x["sheet_thickness_pixcel"]
                rel_error = round(sheet_thickness_pixcel_rev / sheet_thickness_pixcel_0 - 1.0, 3)
                print("- diff_y = [", diff_y, "]")
                print("- sheet_thickness_pixcel_total = [", sheet_thickness_pixcel_total, "]")
                print("- (diff_y - sheet_thickness_pixcel_total) = [", diff_y - sheet_thickness_pixcel_total, "]")
                print("- sheet_thickness_pixcel_0 = [", sheet_thickness_pixcel_0, "]")
                print("- sheet_thickness_pixcel_rev = [", sheet_thickness_pixcel_rev, "]")
                print("- rel_error = [", rel_error, "]")
            for lst_x in lst_sheet_count:
                lst_x["sheet_thickness_pixcel"] = sheet_thickness_pixcel_rev
    else:
        print("sheet_count_check = [", dict_input["sheet_count_check"], "]")
        print("estimate_sheet_count_equally...")
        lst_sheet_count = estimate_sheet_count_equally(img, dict_corners, dict_input)

    if IS_DEBUG_PRINT: 
        print("=== [END] estimate_sheet_count ===\n")

    return dict_corners, lst_sheet_count


# name : get_coordinate_sheet_pair
# brief : 段ボールのシートペア座標の取得
def get_coordinate_sheet_pair(dict_input, dict_corners, lst_sheet_count):

    # get param
    judge_parity = dict_input["judge_parity"]
    Flute = dict_input["Flute"]
    BoxesPerBD = dict_input["BoxesPerBD"]
    multiple_of_sheet_thickness_pixcel = len(Flute)
    lower_part_l = (2 * len(Flute)) # (=2 or 4)
    
    # get param
    #num_of_sheets = int(lst_sheet_count[0]["num_of_sheets"])
    num_of_sheets = 2 * BoxesPerBD
    sheet_thickness_pixcel = int(lst_sheet_count[0]["sheet_thickness_pixcel"])
    min_x = int(lst_sheet_count[0]["min_x"])
    max_x = int(lst_sheet_count[0]["max_x"])
    edge_coord_x = min_x if judge_parity == "left" else max_x

    lst_sheet_pair = [] # init
    pair_id = 0
    l = 0
    for lst_v in lst_sheet_count:
        if lst_v["enable"] == True: # skip hollow part that is not cardboard
            l += 1
            # upper part in sheet pair
            if l == 1:
                pair_id += 1
                y0 = int(lst_v["upper_y"]) # upper line
                y1 = int(lst_v["lower_y"]) # middle line
            # lower part in sheet pair
            elif l == lower_part_l: # lower_part_l (=2 or 4)
                y2 = int(lst_v["lower_y"]) # lower line
                dict_l = {
                    "num_of_sheets": num_of_sheets, 
                    #"sheet_thickness_pixcel": int(multiple_of_sheet_thickness_pixcel * sheet_thickness_pixcel),
                    "sheet_thickness_pixcel": int(sheet_thickness_pixcel),
                    "sheet_pair_id": pair_id,
                    "c11_upper_part_in_upper_sheet_pair":[edge_coord_x, y0],  
                    "c12_lower_part_in_upper_sheet_pair":[edge_coord_x, y1],
                    "c21_upper_part_in_lower_sheet_pair":[edge_coord_x, y1],  
                    "c22_lower_part_in_lower_sheet_pair":[edge_coord_x, y2]
                }
                lst_sheet_pair.append(dict_l)
                l = 0 # reset
    
    return lst_sheet_pair


# name : get_corners_Harris_sub_region_parity
# brief : サブ領域に対するHarrisのコーナー検知
def get_corners_Harris_sub_region_parity(img, judge_parity, dict_sub_region, threshold = 0.001):

    dict_res = {} # init

    # get_corners_Harris_0
    corners = get_corners_Harris_0(img, threshold)
    
    # param_delta_ratio
    param_delta_ratio = 0.3

    # set (numpy.int64)
    c11_upper_y_in_upper_sheet_pair = np.int64(dict_sub_region["c11_upper_part_in_upper_sheet_pair"][1]) # y
    c12_lower_y_in_upper_sheet_pair = np.int64(dict_sub_region["c12_lower_part_in_upper_sheet_pair"][1]) # y
    c21_upper_y_in_lower_sheet_pair = np.int64(dict_sub_region["c21_upper_part_in_lower_sheet_pair"][1]) # y
    c22_lower_y_in_lower_sheet_pair = np.int64(dict_sub_region["c22_lower_part_in_lower_sheet_pair"][1]) # y
    
    if judge_parity == "left":
        # upper part of pair sheets (c11-c12)
        delta_upper = param_delta_ratio * (c12_lower_y_in_upper_sheet_pair - c11_upper_y_in_upper_sheet_pair)
        filFlg_upper = (corners[:,0] >= c11_upper_y_in_upper_sheet_pair + delta_upper) & (corners[:,0] <= c12_lower_y_in_upper_sheet_pair - delta_upper)
        corners_upper = corners[filFlg_upper]
        upper_min_x = int(np.min(corners_upper[:,1])) # min
        d01_detected_edge_x_in_upper_sheet = upper_min_x # min

        # lower part of pair sheets (c21-c22)
        delta_lower = param_delta_ratio * (c22_lower_y_in_lower_sheet_pair - c21_upper_y_in_lower_sheet_pair)
        filFlg_lower = (corners[:,0] >= c21_upper_y_in_lower_sheet_pair + delta_lower) & (corners[:,0] <= c22_lower_y_in_lower_sheet_pair - delta_lower)
        corners_lower = corners[filFlg_lower]
        lower_min_x = int(np.min(corners_lower[:,1])) # min
        d02_detected_edge_x_in_lower_sheet = lower_min_x # min

    elif judge_parity == "right":
        # upper part of pair sheets (c11-c12)
        delta_upper = param_delta_ratio * (c12_lower_y_in_upper_sheet_pair - c11_upper_y_in_upper_sheet_pair)
        filFlg_upper = (corners[:,0] >= c11_upper_y_in_upper_sheet_pair + delta_upper) & (corners[:,0] <= c12_lower_y_in_upper_sheet_pair - delta_upper)
        corners_upper = corners[filFlg_upper]
        upper_max_x = int(np.max(corners_upper[:,1])) # max
        d01_detected_edge_x_in_upper_sheet = upper_max_x # max

        # lower part of pair sheets (c21-c22)
        delta_lower = param_delta_ratio * (c22_lower_y_in_lower_sheet_pair - c21_upper_y_in_lower_sheet_pair)
        filFlg_lower = (corners[:,0] >= c21_upper_y_in_lower_sheet_pair + delta_lower) & (corners[:,0] <= c22_lower_y_in_lower_sheet_pair - delta_lower)
        corners_lower = corners[filFlg_lower]
        lower_max_x = int(np.max(corners_lower[:,1])) # max
        d02_detected_edge_x_in_lower_sheet = lower_max_x #max

    # set
    dict_res = {
        "judge_parity": judge_parity,
        "sheet_pair_id": int(dict_sub_region["sheet_pair_id"]),
        "c11_upper_y_in_upper_sheet_pair": c11_upper_y_in_upper_sheet_pair,
        "c12_lower_y_in_upper_sheet_pair": c12_lower_y_in_upper_sheet_pair,
        "c21_upper_y_in_lower_sheet_pair": c21_upper_y_in_lower_sheet_pair,
        "c22_lower_y_in_lower_sheet_pair": c22_lower_y_in_lower_sheet_pair,
        "coord_xy_upper_sheet": dict_sub_region["c11_upper_part_in_upper_sheet_pair"],
        "coord_xy_lower_sheet": dict_sub_region["c21_upper_part_in_lower_sheet_pair"],
        "d01_detected_edge_x_in_upper_sheet": d01_detected_edge_x_in_upper_sheet,
        "d02_detected_edge_x_in_lower_sheet": d02_detected_edge_x_in_lower_sheet
    }
    
    return dict_res


# name : calc_edge_difference
# brief : Edge差分の計算
# [input] : dict_corners_sub_region_
# [input] : dict_param_
# [output] : dict_edge_result
def calc_edge_difference(dict_corners_sub_region_, dict_param_):

    # get param
    sheet_pair_id = int(dict_corners_sub_region_["sheet_pair_id"])
    edge_coord_upper_x = float(dict_corners_sub_region_["d01_detected_edge_x_in_upper_sheet"]) # float
    edge_coord_lower_x = float(dict_corners_sub_region_["d02_detected_edge_x_in_lower_sheet"]) # float
    edge_coord_upper_y = float(dict_corners_sub_region_["c11_upper_y_in_upper_sheet_pair"]) # float
    edge_coord_lower_y = float(dict_corners_sub_region_["c21_upper_y_in_lower_sheet_pair"]) # float

    c11_upper_y_in_upper_sheet_pair = float(dict_corners_sub_region_["c11_upper_y_in_upper_sheet_pair"]) # float
    c12_lower_y_in_upper_sheet_pair = float(dict_corners_sub_region_["c12_lower_y_in_upper_sheet_pair"]) # float
    c21_upper_y_in_lower_sheet_pair = float(dict_corners_sub_region_["c21_upper_y_in_lower_sheet_pair"]) # float
    c22_lower_y_in_lower_sheet_pair = float(dict_corners_sub_region_["c22_lower_y_in_lower_sheet_pair"]) # float

    # get param
    judge_parity = dict_param_["judge_parity"]
    edge_max_tolerance_mm = dict_param_["edge_max_tolerance_mm"]
    sheet_thickness_mm = dict_param_["sheet_thickness_mm"]
    sheet_thickness_pixcel = dict_param_["sheet_thickness_pixcel"]
    convert_coeff_to_pixcel = dict_param_["convert_coeff_to_pixcel"]
    convert_coeff_to_mm = dict_param_["convert_coeff_to_mm"]
    edge_diff_threshold = dict_param_["edge_diff_threshold"]

    if judge_parity == "left":
        sign_judge_parity = -1 # 左向きのため符号を反転させる
    elif judge_parity == "right":
        sign_judge_parity = 1

    edge_diff = sign_judge_parity * (edge_coord_upper_x - edge_coord_lower_x)
    edge_diff_mm = round((edge_diff * convert_coeff_to_mm), 2)

    if edge_diff > edge_diff_threshold:
        judge_label = EDGE_LABEL_01_INWARD_FOLD # 内折れ
    elif edge_diff < -edge_diff_threshold:
        judge_label = EDGE_LABEL_02_OUTWARD_FOLD # 外折れ
    else: 
        judge_label = EDGE_LABEL_00_NORMAL # 普通

    # set param
    dict_edge_result = { 
        "judge_parity": judge_parity, 
        "sheet_pair_id": sheet_pair_id,
        "edge": judge_label, 
        "edge_diff_mm": edge_diff_mm, 
        "edge_diff_threshold_mm": edge_max_tolerance_mm,
        "edge_diff": edge_diff,
        "edge_diff_threshold": edge_diff_threshold,
        "sheet_thickness_pixcel": sheet_thickness_pixcel,
        "sheet_thickness_mm": sheet_thickness_mm,
        "convert_coeff_to_pixcel": convert_coeff_to_pixcel,
        "edge_coord_upper_x": edge_coord_upper_x,
        "edge_coord_lower_x": edge_coord_lower_x,
        "edge_coord_upper_y": edge_coord_upper_y,
        "edge_coord_lower_y": edge_coord_lower_y,
        "c11_upper_y_in_upper_sheet_pair": c11_upper_y_in_upper_sheet_pair,
        "c12_lower_y_in_upper_sheet_pair": c12_lower_y_in_upper_sheet_pair,
        "c21_upper_y_in_lower_sheet_pair": c21_upper_y_in_lower_sheet_pair,
        "c22_lower_y_in_lower_sheet_pair": c22_lower_y_in_lower_sheet_pair
    }

    return dict_edge_result


# name : judge_edge_status
# brief : Edge差分値に基づいた状態判定
def judge_edge_status(config, lst_details):
    """
    # 内折れしているシートと外折れしているシートが共に3枚/10(30%)以上の時は「外折れ、内折れが混ざる」(優先度1)
    # 内折れしているシートが5枚/10(50%)以上の時は「全体的に内折れ」(優先度2)
    # 外折れしているシートが5枚/10(50%)以上の時は「全体的に外折れ」(優先度3)
    # 内折れしているシートが3枚/10(30%)以上の時は「内折れ、普通が混ざる」(優先度4)
    # 外折れしているシートが3枚/10(30%)以上の時は「外折れ、普通が混ざる」(優先度5)
    # 上記以外:「普通」
    """

    # edge
    count_normal = 0
    count_inward = 0
    count_outward = 0
    for d_res in lst_details:
        if d_res["edge"] == EDGE_LABEL_00_NORMAL: count_normal += 1
        if d_res["edge"] == EDGE_LABEL_01_INWARD_FOLD: count_inward += 1
        if d_res["edge"] == EDGE_LABEL_02_OUTWARD_FOLD: count_outward += 1
    count_total = count_normal + count_inward + count_outward
    rate_inward = count_inward / count_total
    rate_outward = count_outward / count_total
    
    if rate_inward >= 0.3 and rate_outward >= 0.3:
        judged_status = "01_MIX_INWARDS_AND_OUTWARDS"
    elif rate_inward >= 0.5:
        judged_status = "02_OVERALL_INWARDS"
    elif rate_outward >= 0.5:
        judged_status = "03_OVERALL_OUTWARDS"
    elif rate_inward >= 0.3:
        judged_status = "04_MIX_INWARDS_AND_NORMAL"
    elif rate_outward >= 0.3:
        judged_status = "05_MIX_OUTWARDS_AND_NORMAL"
    else:
        judged_status = "06_NORMAL"

    judge_result = config["EDGE_STATUS"]["EN"][judged_status]
    judge_result_ja = config["EDGE_STATUS"]["JA"][judged_status]    

    # set
    dict_res = {
        "00_NORMAL": count_normal,
        "01_INWARD_FOLD": count_inward,
        "02_OUTWARD_FOLD": count_outward,
        "03_TOTAL": count_total,
        "judge_result": judge_result,
        "judge_result_ja": judge_result_ja   
    }
   
    return dict_res


# name : edge_folding_check
# brief : 内折れ/外折れ判定モジュール
def edge_folding_check(img, dict_input):   
    """
    - edge_folding_check() : this module
        - remove_background() : preprocess
        - estimate_sheet_count()
            - for each detected line group
                - estimate_sheet_count_trial()
        - get_coordinate_sheet_pair()
            - for each sheet_pair
                - get_corners_Harris_sub_region_parity()
                - calc_edge_difference()
        - judge_edge_status() -> dict_res[output]
    """

    # set
    Flute = str(dict_input["Flute"])
    Flute_LABEL = "FLUTE_"+str(Flute).upper()
    judge_parity = str(dict_input["judge_parity"])
    BoxesPerBD = int(dict_input["BoxesPerBD"])
    num_of_sheets = 2 * BoxesPerBD
    is_output_annotated_image = bool(dict_input["is_output_annotated_image"])
    lst_annotation_types = dict_input["annotation_type"].split('|')
    lst_annotation_types.sort()

    # copy
    img_org = img.copy()

    # preprocess
    img = preprocess(img)

    # estimate_sheet_count
    dict_corners, lst_sheet_count = estimate_sheet_count(img, dict_input)

    # get_coordinate_sheet_pair
    lst_sheet_pair = get_coordinate_sheet_pair(dict_input, dict_corners, lst_sheet_count)

    lst_details = [] # init
    for l, dict_sub_region in enumerate(lst_sheet_pair):
 
        # get_corners_Harris_sub_region
        threshold = 0.0001
        dict_corners_sub_region = get_corners_Harris_sub_region_parity(img, judge_parity, dict_sub_region, threshold)
        
        # set parameters
        edge_max_tolerance_mm = float(config["EDGE_MAXIMUM_TOLERANCE_MILLIMETER"][Flute_LABEL]) # mm単位
        sheet_thickness_mm = float(config["SHEET_THICKNESS_MILLIMETER"][Flute_LABEL]) # mm単位
        sheet_thickness_pixcel = float(lst_sheet_pair[0]["sheet_thickness_pixcel"])  # 画像検知結果から取得(pixcel単位)
        convert_coeff_to_pixcel = round(sheet_thickness_pixcel / sheet_thickness_mm, 3)
        convert_coeff_to_mm = float(sheet_thickness_mm / sheet_thickness_pixcel)
        edge_diff_threshold = round(edge_max_tolerance_mm * convert_coeff_to_pixcel, 2)
        
        # set parameters for judgement
        dict_param = {
            "judge_parity": judge_parity,
            "edge_max_tolerance_mm": edge_max_tolerance_mm,
            "sheet_thickness_mm": sheet_thickness_mm,
            "sheet_thickness_pixcel": sheet_thickness_pixcel,
            "convert_coeff_to_pixcel": convert_coeff_to_pixcel,
            "convert_coeff_to_mm": convert_coeff_to_mm,
            "edge_diff_threshold": edge_diff_threshold
        }

        # calc_edge_difference
        dict_edge_result = calc_edge_difference(dict_corners_sub_region, dict_param)
        
        # append
        lst_details.append(dict_edge_result)

    # judge_edge_status
    dict_edge_summary = judge_edge_status(config, lst_details)
 
    # output annotated image
    dict_output_base64_image = {} # init
    if is_output_annotated_image:
        df_check_result = pd.DataFrame(lst_details)
        df_check_result.sort_values(by=['sheet_pair_id'], ascending=[True], inplace=True) # sort
        for annotation_type in lst_annotation_types:
            if annotation_type == "1_edge_dots":
                img_00 = draw_edge_from_judge_result(img_org, df_check_result)
                dict_output_base64_image[annotation_type] = cv_to_base64(img_00)
            elif annotation_type == "2_segmented_lines":
                img_01 = draw_sheet_separated_line(img_org, lst_sheet_count)
                dict_output_base64_image[annotation_type] = cv_to_base64(img_01)
            elif annotation_type == "3_edge_dots_and_segmented_lines":
                img_01 = draw_sheet_separated_line(img_org, lst_sheet_count)
                img_02 = draw_edge_from_judge_result(img_01, df_check_result)
                dict_output_base64_image[annotation_type] = cv_to_base64(img_02)

    # dict_res
    dict_res = {
        "input_condition": {}, # init
        "edge_summary": dict_edge_summary, 
        "edge_details": lst_details,
        "sheet_count": lst_sheet_count,
        "dict_output_base64_image": dict_output_base64_image 
    }

    return dict_res


# name : calc_black_ratio
# brief : N分割したbinごとに黒色の比率を計算する
def calc_black_ratio(img):
    
    #n_bins = 25
    n_bins = 30

    height, width, _ = img.shape

    # 黒色の定義（ここではRGBの各成分が50未満）
    black_threshold = 50

    # binの幅
    bin_width = width // n_bins
    
    # init
    df = pd.DataFrame(
        index=[i for i in range(n_bins)], 
        columns=['segment_id', 'start_x', 'end_x', 'black_ratio']
    )

    for i in range(n_bins):
        # セグメントの開始と終了位置を計算
        start_x = int(i * bin_width)
        end_x = int((i + 1) * bin_width) if (i + 1) * bin_width <= width else width
        # 各行のこのセグメントの黒色ピクセル数をカウント
        black_pixel_count = 0
        total_pixels = 0

        for y in range(height):
            row = img[y, start_x:end_x]
            black_pixels = np.sum(np.all(row < black_threshold, axis=-1))
            black_pixel_count += black_pixels
            total_pixels += (end_x - start_x)

        # 黒色率を計算
        black_ratio = round(black_pixel_count / total_pixels, 3) if total_pixels > 0 else 0

        # set
        df.loc[i,"segment_id"] = i+1
        df.loc[i,"start_x"] = start_x
        df.loc[i,"end_x"] = end_x
        df.loc[i,"black_ratio"] = black_ratio

    df["n_bins"] = n_bins
    df["max_flag"] = 0 # init

    max_row_value = df['black_ratio'].max()
    max_row_index = df['black_ratio'] == max_row_value
    df.loc[max_row_index, "max_flag"] = 1

    df_all = df.copy()

    # filter
    fillFlg = (df["max_flag"] == 1)
    df = df[fillFlg].reset_index(drop=True)
    lower_x = int(df["start_x"].tolist()[0])
    upper_x = int(df["end_x"].tolist()[0])
    mid_x = int(0.5 * (lower_x + upper_x))
    w = (upper_x - lower_x)

    #multiple_ratio = 1.5
    multiple_ratio = (n_bins / 20)
    #multiple_ratio = (n_bins / 15)

    range_lower_x = int(mid_x - multiple_ratio * w)
    range_upper_x = int(mid_x + multiple_ratio * w)

    # dict_res
    dict_res = {
        "height": height,
        "width": width,
        "n_bins": n_bins, 
        "bin_width": bin_width,
        "lower_x": range_lower_x,
        "upper_x": range_upper_x,
        "mid_x": mid_x
    }

    return dict_res


# name : estimate_gap_length_by_cornerHarris_sub_region
# brief : 検知されたエッジ座標からgap箇所を特定し長さを推定する
def estimate_gap_length_by_cornerHarris_sub_region(img, dict_sub_region, threshold):

    # fillConvexPoly
    ##LOWER_AREA_RATIO = 0.001
    #LOWER_AREA_RATIO = 0.01
    LOWER_AREA_RATIO = 0.05
    img_filled = fillConvexPoly(img, LOWER_AREA_RATIO, COLOR_LIME)

    # calc_black_ratio
    dict_center_x = calc_black_ratio(img_filled)
    
    # get_corners_Harris_0
    corners = get_corners_Harris_0(img_filled, threshold)

    # set (numpy.int64)
    c11_upper_y_in_upper_sheet_pair = np.int64(dict_sub_region["c11_upper_part_in_upper_sheet_pair"][1]) # y
    c12_lower_y_in_upper_sheet_pair = np.int64(dict_sub_region["c12_lower_part_in_upper_sheet_pair"][1]) # y
    c21_upper_y_in_lower_sheet_pair = np.int64(dict_sub_region["c21_upper_part_in_lower_sheet_pair"][1]) # y
    c22_lower_y_in_lower_sheet_pair = np.int64(dict_sub_region["c22_lower_part_in_lower_sheet_pair"][1]) # y
    
    # upper part of pair sheets
    #filFlg_upper = (corners[:,0] >= c11_upper_y_in_upper_sheet_pair) & (corners[:,0] <= c12_lower_y_in_upper_sheet_pair)
    #corners_upper = corners[filFlg_upper]
    #dict_estimated_gap_upper = estimate_gap_length(corners_upper, dict_center_x)
    
    # lower part of pair sheets
    filFlg_lower = (corners[:,0] >= c21_upper_y_in_lower_sheet_pair) & (corners[:,0] <= c22_lower_y_in_lower_sheet_pair)
    corners_lower = corners[filFlg_lower]
    dict_estimated_gap_lower = estimate_gap_length(corners_lower, dict_center_x)
    
    # set
    dict_res = {
        "sheet_pair_id": int(dict_sub_region["sheet_pair_id"]),
        #"estimated_upper_left_x": dict_estimated_gap_upper["estimated_left_x"],
        #"estimated_upper_right_x": dict_estimated_gap_upper["estimated_right_x"],
        #"estimated_gap_upper": dict_estimated_gap_upper["estimated_gap"],         
        "estimated_lower_left_x": safe_cast_to_integer(dict_estimated_gap_lower["estimated_left_x"]),
        "estimated_lower_right_x": safe_cast_to_integer(dict_estimated_gap_lower["estimated_right_x"]),
        "estimated_gap_lower": safe_cast_to_integer(dict_estimated_gap_lower["estimated_gap"]),
        "estimated_gap_center_x": safe_cast_to_integer(dict_center_x["mid_x"]), 
        "c21_upper_y_in_lower_sheet_pair": c21_upper_y_in_lower_sheet_pair,
        "c22_lower_y_in_lower_sheet_pair": c22_lower_y_in_lower_sheet_pair
    }

    return dict_res


# name : estimate_gap_length
# brief : 検知されたエッジ座標からのgap長さの推定
def estimate_gap_length(coords, dict_center_x):

    try:
        lower_x = int(dict_center_x["lower_x"])
        upper_x = int(dict_center_x["upper_x"])
        mid_x = int(dict_center_x["mid_x"])

        # init
        dict_res = {
            "estimated_left_x": 0,
            "estimated_right_x": 0,
            "estimated_gap": 0
        }

        x_coords = coords[:,1].tolist()
        x_coords_left = [x for x in x_coords if lower_x <= x <= mid_x]
        x_coords_right = [x for x in x_coords if mid_x < x <= upper_x]
        df_freq_left = vec_to_df_freq(x_coords_left)
        df_freq_right = vec_to_df_freq(x_coords_right)
        estimated_left_x = 0
        estimated_right_x = 0
        estimated_gap = 0
        if len(df_freq_left) > 0:
            estimated_left_x = safe_cast_to_integer(df_freq_left.loc[0,"Item"])
        if len(df_freq_right) > 0:
            estimated_right_x = safe_cast_to_integer(df_freq_right.loc[0,"Item"])
        if estimated_left_x > 0 and estimated_right_x > 0:
            estimated_gap = max(estimated_right_x - estimated_left_x, 0)
        
        # set
        dict_res["estimated_left_x"] = estimated_left_x
        dict_res["estimated_right_x"] = estimated_right_x
        dict_res["estimated_gap"] = estimated_gap
    
    except:
        return dict_res
    
    return dict_res


# name : judge_gap_length
def judge_gap_length(dict_corners_sub_region, dict_param):
    
    # get param
    sheet_pair_id = int(dict_corners_sub_region["sheet_pair_id"])
    estimated_gap = dict_corners_sub_region["estimated_gap_lower"] # lower (ジョイントは必ず下側で行うため、ギャップは常にシートの下側になる) 
    lowerx = int(dict_param["gap_lower_threshold"])
    upperx = int(dict_param["gap_upper_threshold"])
    convert_coeff_to_mm = float(dict_param["convert_coeff_to_mm"])
    
    # init
    judged_result = "."
    judged_result_ja = "."
    estimated_gap_mm = 0.0
    if estimated_gap is not None:
        estimated_gap_mm = round(estimated_gap * convert_coeff_to_mm, 2)
        if estimated_gap >= lowerx and estimated_gap <= upperx:
            judged_label = "00_NORMAL"
        elif estimated_gap < lowerx:
            judged_label = "01_NARROW"
        elif estimated_gap > upperx:
            judged_label = "02_WIDE"
        
        judged_result = config["GAP_LABEL"]["EN"][judged_label]
        judged_result_ja = config["GAP_LABEL"]["JA"][judged_label]
    
    # set
    dict_res = {
        "sheet_pair_id": sheet_pair_id,
        "estimated_gap_mm": estimated_gap_mm, 
        "gap_lower_threshold_mm": dict_param["gap_lower_threshold_mm"],
        "gap_upper_threshold_mm": dict_param["gap_upper_threshold_mm"],
        "judge_result": judged_result,
        "judge_result_ja": judged_result_ja,
        "estimated_gap_pixcel": estimated_gap,
        "gap_lower_threshold_pixcel": lowerx,
        "gap_upper_threshold_pixcel": upperx,
        "sheet_thickness_pixcel": dict_param["sheet_thickness_pixcel"],
        "sheet_thickness_mm": dict_param["sheet_thickness_mm"],
        "estimated_gap_center_x": int(dict_corners_sub_region["estimated_gap_center_x"]),
        "estimated_lower_left_x": int(dict_corners_sub_region["estimated_lower_left_x"]),
        "estimated_lower_right_x": int(dict_corners_sub_region["estimated_lower_right_x"]),
        "c21_upper_y_in_lower_sheet_pair": int(dict_corners_sub_region["c21_upper_y_in_lower_sheet_pair"]),
        "c22_lower_y_in_lower_sheet_pair": int(dict_corners_sub_region["c22_lower_y_in_lower_sheet_pair"])
    } 

    return dict_res


# name : judge_gap_status
# brief : gap推定値に基づいた状態判定
def judge_gap_status(config, lst_details):
    """
    # ①「狭い」と「広い」が共に30%以上ある場合、「狭い、広いが混ざる」
    # ②「狭い」が50%以上ある場合、「全体的に狭い」
    # ③「広い」が50%以上ある場合、「全体的に広い」
    # ④「狭い」が30%以上ある場合、「狭い、普通が混ざる」
    # ⑤「広い」が30%以上ある場合、「広い、普通が混ざる」
    # ⑥「狭い」と「広い」が共に30%未満の場合、「全体的に普通」
    """

    # init
    count_00_normal = 0
    count_01_narrow = 0
    count_02_wide = 0

    for d in lst_details:
        if d["judge_result"] == config["GAP_LABEL"]["EN"]["00_NORMAL"]: count_00_normal += 1
        if d["judge_result"] == config["GAP_LABEL"]["EN"]["01_NARROW"]: count_01_narrow += 1
        if d["judge_result"] == config["GAP_LABEL"]["EN"]["02_WIDE"]: count_02_wide += 1
    
    count_total = count_00_normal + count_01_narrow + count_02_wide
    rate_00_normal = round(count_00_normal / count_total, 3)
    rate_01_narrow = round(count_01_narrow / count_total, 3)
    rate_02_wide = round(count_02_wide / count_total, 3)
    
    # init
    judge_result = "."
    judge_result_ja = "."

    if rate_01_narrow >= 0.3 and rate_02_wide >= 0.3:
        judged_status = "01_MIX_NARROW_AND_WIDE"
    elif rate_01_narrow >= 0.5:
        judged_status = "02_OVERALL_NARROW"
    elif rate_02_wide >= 0.5:
        judged_status = "03_OVERALL_WIDE"
    elif rate_01_narrow >= 0.3 and rate_01_narrow < 0.5:
        judged_status = "04_MIX_NARROW_AND_NORMAL"
    elif rate_02_wide >= 0.3 and rate_02_wide < 0.5:
        judged_status = "05_MIX_WIDE_AND_NORMAL"
    else:
        judged_status = "06_NORMAL"
    
    judge_result = config["GAP_STATUS"]["EN"][judged_status]
    judge_result_ja = config["GAP_STATUS"]["JA"][judged_status]    

    # set
    dict_res = {
        "count_00_normal": count_00_normal,
        "count_01_narrow": count_01_narrow,
        "count_02_wide": count_02_wide,
        "count_total": count_total,
        "judge_result": judge_result,
        "judge_result_ja": judge_result_ja   
    }

    return dict_res



# name : gap_size_check
# brief : ギャップ判定モジュール
def gap_size_check(img, dict_input):
    """
    - gap_size_check() : this module
        - remove_background() : preprocess
        - estimate_sheet_count()
            - for each detected line group
                - estimate_sheet_count_trial()
        - get_coordinate_sheet_pair()
            - for each sheet_pair
                - get_corners_Harris_sub_region_parity()
                - calc_edge_difference()
        - judge_gap_status() -> dict_res[output]
    """

    # set
    Flute = str(dict_input["Flute"])
    Flute_LABEL = "FLUTE_"+str(Flute).upper()
    judge_parity = str(dict_input["judge_parity"])
    BoxesPerBD = int(dict_input["BoxesPerBD"])
    num_of_sheets = 2 * BoxesPerBD
    is_output_annotated_image = bool(dict_input["is_output_annotated_image"])
    lst_annotation_types = dict_input["annotation_type"].split('|')
    lst_annotation_types.sort()

    # copy
    img_org = img.copy()

    # preprocess
    img = preprocess(img)

    # estimate_sheet_count
    dict_corners, lst_sheet_count = estimate_sheet_count(img, dict_input)

    # get_coordinate_sheet_pair
    lst_sheet_pair = get_coordinate_sheet_pair(dict_input, dict_corners, lst_sheet_count)

    lst_details = [] # init
    for l, dict_sub_region in enumerate(lst_sheet_pair):
 
        # get_corners_Harris_sub_region
        threshold = 0.000001
        dict_corners_sub_region = estimate_gap_length_by_cornerHarris_sub_region(img, dict_sub_region, threshold)
        if False:
            pprint(dict_corners_sub_region)
            print("\n\n")

        # get param
        sheet_thickness_pixcel = float(lst_sheet_pair[0]["sheet_thickness_pixcel"])  # 画像検知結果から取得(pixcel単位)
        sheet_thickness_mm = float(config["SHEET_THICKNESS_MILLIMETER"][Flute_LABEL]) # mm単位
        gap_lower_threshold_mm = config["GAP_SIZE_CHECK"]["GAP_SIZE_LOWER_THREHOLD_MILLIMETER"] # mm単位
        gap_upper_threshold_mm = config["GAP_SIZE_CHECK"]["GAP_SIZE_UPPER_THREHOLD_MILLIMETER"] # mm単位
        gap_lower_threshold = int(round(gap_lower_threshold_mm / sheet_thickness_mm * sheet_thickness_pixcel,0))
        gap_upper_threshold = int(round(gap_upper_threshold_mm / sheet_thickness_mm * sheet_thickness_pixcel,0))
        convert_coeff_to_pixcel = round(sheet_thickness_pixcel / sheet_thickness_mm, 3)
        convert_coeff_to_mm = float(sheet_thickness_mm / sheet_thickness_pixcel)
        
        # set parameters for judgement
        dict_param = {
            "sheet_thickness_pixcel": sheet_thickness_pixcel,
            "sheet_thickness_mm": sheet_thickness_mm,
            "gap_lower_threshold_mm": gap_lower_threshold_mm,
            "gap_upper_threshold_mm": gap_upper_threshold_mm,
            "gap_lower_threshold": gap_lower_threshold,
            "gap_upper_threshold": gap_upper_threshold,
            "convert_coeff_to_pixcel": convert_coeff_to_pixcel,
            "convert_coeff_to_mm": convert_coeff_to_mm
        }
    
        # judge_gap_length
        dict_gap_result = judge_gap_length(dict_corners_sub_region, dict_param)
        
        # append
        lst_details.append(dict_gap_result)

    # judge_gap_status
    dict_gap_summary = judge_gap_status(config, lst_details)
    
     # output annotated image
    dict_output_base64_image = {} # init
    if is_output_annotated_image:
        df_check_result = pd.DataFrame(lst_details)
        df_check_result.sort_values(by=['sheet_pair_id'], ascending=[True], inplace=True) # sort
        for annotation_type in lst_annotation_types:
            if annotation_type == "1_edge_dots":
                img_00 = draw_gap_size_from_judge_result(img_org, df_check_result)
                dict_output_base64_image[annotation_type] = cv_to_base64(img_00)
            elif annotation_type == "2_segmented_lines":
                img_01 = draw_sheet_separated_line(img_org, lst_sheet_count)
                dict_output_base64_image[annotation_type] = cv_to_base64(img_01)
            elif annotation_type == "3_edge_dots_and_segmented_lines":
                img_01 = draw_sheet_separated_line(img_org, lst_sheet_count)
                img_02 = draw_gap_size_from_judge_result(img_01, df_check_result)
                dict_output_base64_image[annotation_type] = cv_to_base64(img_02)

    # dict_res
    dict_res = {
        "input_condition": {}, # init
        "gap_summary": dict_gap_summary, 
        "gap_details": lst_details, 
        "sheet_count": lst_sheet_count,
        "dict_output_base64_image": dict_output_base64_image  
    }

    return dict_res


# name : judge_sheet_condition
# brief : シート状態判定のメインモジュール
def judge_sheet_condition(img, dict_input):

    h, w, _ = img.shape
    
    # datetime
    start_tdatetime = dt.now(timezone('Asia/Tokyo'))
    start_datetime_str_output = start_tdatetime.strftime('%Y-%m-%d %H:%M:%S')

    # get test name
    test_name = dict_input["test_name"]
    test_name_ja = config["TEST_NAME_OF_SHEET_CONDITION"]["JA"][test_name]
    
    #===================
    #=== main module ===
    #===================

    # edge_folding_check
    if test_name == TEST_01_EDGE_FOLDING_CHECK:
        dict_res = edge_folding_check(img, dict_input)
    # gap_size_check
    elif test_name == TEST_02_JOINT_GAP_SIZE_CHECK:
        dict_res = gap_size_check(img, dict_input)

    lst_sheet_count = dict_res["sheet_count"]
    
    # datetime
    end_tdatetime = dt.now(timezone('Asia/Tokyo'))
    end_datetime_str_output = end_tdatetime.strftime('%Y-%m-%d %H:%M:%S')

    # add conditions
    dict_res["input_condition"] = {
        "test_name": test_name,
        "test_name_ja": test_name_ja,
        "image_width_pixcel": int(w),
        "image_height_pixcel": int(h),
        "Flute": dict_input["Flute"],
        "BoxesPerBD": dict_input["BoxesPerBD"], 
        "judge_parity": dict_input["judge_parity"],
        "start_datetime": start_datetime_str_output,
        "end_datetime": end_datetime_str_output,
        "noise_removal_filter": str(config["NOISE_REMOVAL_FILTER_TYPE"]),
        "logic_last_updated": str(config["LOGIC_LAST_UPDATED"])
    }

    return dict_res