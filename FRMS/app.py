"""
FRMS/app.py

Copyright (C) 2021 Дмитрий Кузнецов

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import io
from PIL import Image
from FRMS.utils.face_detector import FaceDetector
from FRMS.utils.feature_extractor import FeatureExtractor
from FRMS.utils.feature_matcher import FeatureMatcher
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
    THRESHOLD = float(os.environ['THRESHOLD'])
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
