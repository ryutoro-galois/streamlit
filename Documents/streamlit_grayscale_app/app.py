import streamlit as st
from PIL import Image
import io

# Streamlitアプリのセットアップ
st.title("画像のグレースケール変換アプリ")
uploaded_file = st.file_uploader("画像をアップロードしてください (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像のグレースケール変換
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    grayscale_image = image.convert("L")
    st.image(grayscale_image, caption="グレースケール画像", use_column_width=True)

    # ダウンロードボタンを追加
    buf = io.BytesIO()
    grayscale_image.save(buf, format="PNG")
    st.download_button(
        label="グレースケール画像をダウンロード",
        data=buf.getvalue(),
        file_name="grayscale_image.png",
        mime="image/png"
    )

# ポートとアドレスの明確な指定 (重要!!)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # 環境変数PORTを使用。デフォルト8080
    st._config.set_option('server.port', port)  # ポート設定
    st._config.set_option('server.address', '0.0.0.0')  # 外部からもアクセス可能にする