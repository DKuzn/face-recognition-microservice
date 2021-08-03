"""
FRMS/utils/face_detector.py

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

from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from PIL.Image import Image
from typing import List, Tuple
import torch


class FaceDetector:
    def __init__(self, img_size: int = 160, min_face_size: int = 20):
        self._img_size = img_size
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mtcnn = MTCNN(
            image_size=self._img_size, margin=0, min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self._device
        )

    def find_faces(self, img: Image) -> List[Tuple[torch.Tensor, List[int]]]:
        bboxes, _ = self._mtcnn.detect(img, landmarks=False)
        faces_and_bboxes = []
        if bboxes is not None:
            for bb in bboxes:
                face = extract_face(img, bb, image_size=self._img_size)
                faces_and_bboxes.append((fixed_image_standardization(face), list(bb)))
        return faces_and_bboxes
