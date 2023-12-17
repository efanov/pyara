#установка корневого образа
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
#FROM nvidia/cuda:11.5.2-cudnn8-runtime-ubuntu18.04
#FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y python3 python3-pip sudo

MAINTAINER Ilia Mironov 'ilyamironov210202@gmail.com'

#создание рабочей директории
WORKDIR /app

COPY requirements.txt requirements.txt

#устанавливает все зависимости
RUN pip3 install -r requirements.txt

#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


#Копируем все содержимое из директории в которой мы щас в ./app
COPY . .

CMD ["python3", "main.py"]