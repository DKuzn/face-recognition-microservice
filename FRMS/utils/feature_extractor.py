# FRMS/utils/feature_extractor.py
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
"""This module contains class FeatureExtractor.

InceptionResnetV1 uses for feature extraction.
"""

from facenet_pytorch import InceptionResnetV1
import torch


class FeatureExtractor:
    """Class for feature extraction.

    Example:
        >>> import torch
        >>> from FRMS.utils.feature_extractor import FeatureExtractor
        >>> img = torch.rand((3, 160, 160))
        >>> feature_extractor = FeatureExtractor()
        >>> features = feature_extractor.extract_features(img)
    """
    def __init__(self):
        self._device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._resnet: InceptionResnetV1 = InceptionResnetV1(pretrained='vggface2').eval().to(self._device)

    def extract_features(self, face: torch.Tensor) -> torch.Tensor:
        """Extract features from given face image tensor.

        Args:
            face: Face image tensor.

        Return:
            Tensor of features.
        """
        face: torch.Tensor = face.to(self._device)
        features: torch.Tensor = self._resnet(face.unsqueeze(0)).detach().cpu()
        return features[0]
