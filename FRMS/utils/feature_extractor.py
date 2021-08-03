"""
FRMS/utils/feature_extractor.py

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

from facenet_pytorch import InceptionResnetV1
import torch


class FeatureExtractor:
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self._device)

    def extract_features(self, face: torch.Tensor) -> torch.Tensor:
        face = face.to(self._device)
        features = self._resnet(face.unsqueeze(0)).detach().cpu()
        return features[0]
