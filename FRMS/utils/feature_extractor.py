# FRMS/utils/feature_extractor.py
#
# Copyright (C) 2021 Дмитрий Кузнецов
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

        Returns:
            Tensor of features.
        """
        face: torch.Tensor = face.to(self._device)
        features: torch.Tensor = self._resnet(face.unsqueeze(0)).detach().cpu()
        return features[0]
