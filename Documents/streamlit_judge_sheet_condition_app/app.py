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
    st.image(image, caption="input image", use_container_width=False)
    img_cv = convert_PIL_to_cv2(image)

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

    # NumPy配列 (OpenCV形式) をPillow形式に変換
    output_image_PIL = Image.fromarray(cv.cvtColor(output_image, cv.COLOR_BGR2RGB))

    st.image(output_image_PIL, caption="output image", use_container_width=False)

    # ダウンロードボタンを追加
    buf = io.BytesIO()
    output_image_PIL.save(buf, format="PNG")
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