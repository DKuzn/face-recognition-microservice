# FRMS/utils/face_detector.py
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
"""This module contains class FaceDetector.

MTCNN uses for face detection.
"""

from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from PIL.Image import Image
from typing import List, Tuple
import torch


class FaceDetector:
    """Class for face detection.

    Args:
        img_size: Size in pixels of cropped face image.
        min_face_size: Size in pixels of minimal face on image.

    Example:
        >>> import PIL
        >>> from FRMS.utils.face_detector import FaceDetector
        >>> detector = FaceDetector()
        >>> img = PIL.Image.open('path/to/img').convert('RGB')
        >>> faces = detector.find_faces(img)
    """
    def __init__(self, img_size: int = 160, min_face_size: int = 20) -> None:
        self._img_size: int = img_size
        self._device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mtcnn: MTCNN = MTCNN(
            image_size=self._img_size, margin=0, min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self._device
        )

    def find_faces(self, img: Image) -> List[Tuple[torch.Tensor, List[int]]]:
        """Find faces on given image.

        Args:
            img: PIL Image.

        Return:
            List of tuples image -- tensor and list of bounding box coordinates.
        """
        bboxes, _ = self._mtcnn.detect(img, landmarks=False)
        faces_and_bboxes: List[Tuple[torch.Tensor, List[int]]] = []
        if bboxes is not None:
            for bb in bboxes:
                face: torch.Tensor = extract_face(img, bb, image_size=self._img_size)
                faces_and_bboxes.append((fixed_image_standardization(face), list(bb)))
        return faces_and_bboxes
