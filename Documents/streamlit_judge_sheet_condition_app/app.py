# app.py

# import libraries
from utility_functions import *
from base_functions import config, judge_sheet_condition
import streamlit as st
from PIL import Image
import io

import os
import base64
import numpy as np
import cv2 as cv
import datetime
from datetime import datetime as dt
from pytz import timezone
from pprint import pprint


# name : base64_to_cv
def base64_to_cv(image_base64):
    image_bytes = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv.imdecode(np_array, cv.IMREAD_COLOR)
    return image_cv


# Streamlitアプリのセットアップ
st.title("アプリ")
uploaded_file = st.file_uploader("画像をアップロードしてください (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="画像", use_column_width=True)

    # dict_input
    dict_input = {
        "test_name": "EDGE_FOLDING_CHECK", 
        "Flute": "B",
        "BoxesPerBD": 10,
        "judge_parity": "left",
        "is_output_annotated_image": True,
        "annotation_type": "1_edge_dots",
        "is_debug_print": True
    }

    # judge_sheet_condition
    dict_check_result = judge_sheet_condition(image, dict_input)

    output_base64_image = dict_res["dict_output_base64_image"][annotation_type]
    output_image = base64_to_cv(output_base64_image)

    st.image(output_image, caption="画像", use_column_width=True)

    # ダウンロードボタンを追加
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    st.download_button(
        label="画像をダウンロード",
        data=buf.getvalue(),
        file_name="output_image.png",
        mime="image/png"
    )

# ポートとアドレスの明確な指定
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # 環境変数PORTを使用。デフォルト8080
    st._config.set_option('server.port', port)  # ポート設定
    st._config.set_option('server.address', '0.0.0.0')  # 外部からもアクセス可能にする









# app.py


# name : handler
def handler(event, context):

    # get request parameters
    img = base64_to_cv(event['base64_image'])
    test_name = str(event['conditions']['test_name'])
    Flute = str(event['conditions']['Flute'])
    BoxesPerBD = int(event['conditions']['BoxesPerBD'])
    judge_parity = str(event['conditions']['judge_parity'])
    is_output_annotated_image = bool(event['options']['is_output_annotated_image'])
    annotation_type = str(event['options']['annotation_type'])

    # dict_input
    dict_input = {
        "test_name": test_name, 
        "Flute": Flute,
        "BoxesPerBD": BoxesPerBD,
        "judge_parity": judge_parity,
        "is_output_annotated_image": is_output_annotated_image,
        "annotation_type": annotation_type,
        "is_debug_print": config["IS_DEBUG_PRINT"]
    }

    # judge_sheet_condition
    dict_check_result = judge_sheet_condition(img, dict_input)
    
    return dict_check_result