FROM ubuntu:latest

RUN apt-get update
RUN apt install -y python3-dev default-libmysqlclient-dev build-essential
RUN apt install -y pip
WORKDIR /root/
RUN mkdir face-recognition-microservice
WORKDIR /root/face-recognition-microservice
COPY FRMS /root/face-recognition-microservice/FRMS
COPY main.py /root/face-recognition-microservice
COPY requirements.txt /root/face-recognition-microservice
RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]