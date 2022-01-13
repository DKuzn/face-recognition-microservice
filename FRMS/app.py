# FRMS/app.py
#
# Copyright (c) 2021 Дмитрий Кузнецов
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
