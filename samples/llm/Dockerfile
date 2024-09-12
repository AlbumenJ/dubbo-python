FROM tensorflow/tensorflow:latest-gpu

RUN apt update && apt install -y git vim && \
  git clone https://github.com/apache/dubbo-python.git && \
  cd dubbo-python && \
  pip install -r requirements.txt && \
  pip install . && \
  pip install transformers && \
  pip install accelerate && \
  pip install protobuf \

ADD . /app
WORKDIR /app

RUN python /app/setup.py

ENTRYPOINT ["python", "/app/server.py"]