from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import io
from PIL import Image
from FRMS.utils import FaceDetector, FeatureExtractor, FeatureMatcher
from FRMS.datamodels import RequestModel, ResponseModel
import os

app = FastAPI(title='Face Recognition Microservice', version='0.1.0')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    THRESHOLD = os.environ['THRESHOLD']
except KeyError:
    THRESHOLD = 0.03

detector = FaceDetector()
feature_extractor = FeatureExtractor()
feature_matcher = FeatureMatcher(max_distance=THRESHOLD)


@app.post('/', response_model=List[ResponseModel])
async def main(request: RequestModel):
    simg = request.image
    byte_img = base64.b64decode(simg)
    img = Image.open(io.BytesIO(byte_img)).convert('RGB')
    faces = detector.find_faces(img)
    data = []
    for face, bb in faces:
        features = feature_extractor.extract_features(face)
        answer = feature_matcher.match_features(features)
        answer['bbox'] = bb
        data.append(answer)

    return data
