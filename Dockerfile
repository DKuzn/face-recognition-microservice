FROM ubuntu:latest

RUN apt update
RUN apt install -y pip
WORKDIR /root/
RUN mkdir FaceRecognitionMicroservice
WORKDIR /root/FaceRecognitionMicroservice
COPY FRMS /root/FaceRecognitionMicroservice/FRMS
COPY main.py /root/FaceRecognitionMicroservice
COPY requirements.txt /root/FaceRecognitionMicroservice
RUN pip install -r requirements.txt

ENTRYPOINT ["gunicorn", "main:app", "-b 0.0.0.0:80", "-w 4", "-k uvicorn.workers.UvicornWorker", "-t 0"]