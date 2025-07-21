# ベースイメージとして AWS Lambda Python 3.11 を使用
FROM public.ecr.aws/lambda/python:3.11

# 作業ディレクトリを設定
WORKDIR /var/task/

# 必要ファイルを一括でコピー
# ローカルの app.py を Lambda のワークディレクトリ `/var/task/` にコピー
COPY . .

# 確認用にファイル一覧を出力
RUN ls -la /var/task/

# 必要なシステムパッケージのインストール (libgl、gccなど)
RUN yum install -y \
    libglvnd-glx \
    && yum clean all

# pip を最新化
RUN pip install --upgrade pip

# Python ライブラリのインストール
# pip install の実行後にキャッシュをクリアすることで、Dockerイメージのサイズを小さくする
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# numba の JIT を無効化
ENV NUMBA_DISABLE_JIT=1

# タイムゾーン設定
ENV TZ=Asia/Tokyo

# Lambda のハンドラー関数 (app.handler を指定)
CMD ["app.handler"]