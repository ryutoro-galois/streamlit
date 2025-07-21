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
st.title("段ボールシート状態チェック")

# test_name
test_name = st.selectbox('select test name', ['EDGE_FOLDING_CHECK', 'JOINT_GAP_SIZE_CHECK'], index=1)

# Flute
Flute = st.selectbox('select Flute', ['A', 'AB', 'CB', 'B', 'C'], index=None)

# judge_parity
judge_parity = st.selectbox('select judge parity', ['left', 'right'], index=1)

# annotation_type
annotation_type = st.selectbox('select annotation type', ['1_edge_dots', '2_segmented_lines', '3_edge_dots_and_segmented_lines'], index=None)

uploaded_file = st.file_uploader("Please upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    img_pil = rotate_image(img_pil)
    file_name = uploaded_file.name
    file_name_0, file_ext = os.path.splitext(file_name)
    str_caption_input = f"input image: [ {file_name} ]"
    # display uploaded image
    st.image(img_pil, caption=str_caption_input, use_container_width=False)
    # convert cv
    img_cv = convert_PIL_to_cv2(img_pil)

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
    dict_check_result = judge_sheet_condition(img_cv, dict_input)

    annotation_type = "1_edge_dots"
    output_base64_image = dict_check_result["dict_output_base64_image"][annotation_type]
    output_image = base64_to_cv(output_base64_image)

    # datetime
    tdatetime = dt.now(timezone('Asia/Tokyo'))
    datetime_str = tdatetime.strftime('%Y%m%d_%H%M%S')
    output_name = f"{file_name_0}_[{annotation_type}]_{datetime_str}.png"
    str_caption_output = f"output image: [ {output_name} ]"

    # NumPy配列 (OpenCV形式) をPillow形式に変換
    output_image_PIL = Image.fromarray(cv.cvtColor(output_image, cv.COLOR_BGR2RGB))

    st.image(output_image_PIL, caption=str_caption_output, use_container_width=False)

    # ダウンロードボタンを追加
    buf = io.BytesIO()
    output_image_PIL.save(buf, format="PNG")
    st.download_button(
        label="Download image",
        data=buf.getvalue(),
        file_name=output_name,
        mime="image/png"
    )

# ポートとアドレスの明確な指定
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # 環境変数PORTを使用。デフォルト8080
    st._config.set_option('server.port', port)  # ポート設定
    st._config.set_option('server.address', '0.0.0.0')  # 外部からもアクセス可能にする