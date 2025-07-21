# ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# 依存関係ファイルを追加してインストール
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをコンテナにコピー
COPY . .

# ポート設定とENTRYPOINT
ENV PORT=8080
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
