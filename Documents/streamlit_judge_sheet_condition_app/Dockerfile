# Python 3.11の公式イメージをベースにする
FROM python:3.11-slim

# 作業ディレクトリを設定 (アプリケーション用)
WORKDIR /app

# 必要な依存ツールをインストール
RUN apt-get update && apt-get install -y \
    build-essential libssl-dev libffi-dev python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ホスト環境のrequirements.txtをコンテナ内にコピー
COPY requirements.txt .

# pipのアップグレードを実行
RUN pip install --upgrade pip

# requirements.txtに書かれたライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコンテナにコピー (必要に応じて)
COPY . .

# 必要な環境変数を設定
# 例: Streamlitアプリのポート設定など
ENV STREAMLIT_SERVER_PORT=8501

# コンテナ起動時に実行するデフォルトコマンド
#CMD ["streamlit", "run", "app.py"]

# ポート設定とENTRYPOINT
ENV PORT=8080
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
