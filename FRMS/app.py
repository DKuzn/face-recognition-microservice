# FRMS/app.py
#
# Copyright (C) 2021-2022  Дмитрий Кузнецов
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""This module contains FastAPI application.

Example:
    >>> from FRMS.app import app
    >>> import uvicorn
    >>> uvicorn.run(app, host='0.0.0.0', port=5000)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Dict, Union
import base64
import io
from PIL import Image
from torch import Tensor
from FRMS.utils.face_detector import FaceDetector
from FRMS.utils.feature_extractor import FeatureExtractor
from FRMS.utils.feature_matcher import FeatureMatcher
from FRMS.datamodels import RequestModel, ResponseModel
from FRMS import __version__
import os

app: FastAPI = FastAPI(title='Face Recognition Microservice', version=__version__)

origins: List[str] = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    THRESHOLD: float = float(os.environ['THRESHOLD'])
except KeyError:
    THRESHOLD: float = 1.0

detector: FaceDetector = FaceDetector()
feature_extractor: FeatureExtractor = FeatureExtractor()
feature_matcher: FeatureMatcher = FeatureMatcher(max_distance=THRESHOLD)


@app.post('/', response_model=List[ResponseModel])
async def main(request: RequestModel):
    """Main route of microservice.

    Args:
        request: Request in JSON-format.

    Return:
        Response in JSON-format.
    """
    simg: str = request.image
    byte_img: bytes = base64.b64decode(simg)
    img: Image.Image = Image.open(io.BytesIO(byte_img)).convert('RGB')
    faces: List[Tuple[Tensor, List[int]]] = detector.find_faces(img)
    data: List[Dict[str, Union[List[int], int, str]]] = []
    for face, bb in faces:
        features = feature_extractor.extract_features(face)
        answer = feature_matcher.match_features(features)
        answer['bbox'] = bb
        data.append(answer)

    return data
