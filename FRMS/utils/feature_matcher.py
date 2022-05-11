# FRMS/utils/feature_matcher.py
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
"""This module contains class FeatureMatcher and function distance.

Features matches by calculating distance between given features tensor
and features tensors from database.
"""

from FRMS.database import Session, Face
from sqlalchemy.orm import Query
from sqlalchemy.exc import InvalidRequestError
from typing import Dict, Union, List, Optional
import torch


class FeatureMatcher:
    """Class for feature matching.

    Args:
        max_distance: Max distance between features.

    Attributes:
        max_distance: Max distance between features,
            if distance higher the this value, face is
            unrecognized.

    Example:
        >>> import torch
        >>> from FRMS.utils.feature_matcher import FeatureMatcher
        >>> features = torch.rand(512)
        >>> feature_matcher = FeatureMatcher(max_distance=1.0)
        >>> result = feature_matcher.match_features(features)
    """
    def __init__(self, max_distance: float = 0.03):
        self.max_distance: float = max_distance
        self._session: Session = Session()

    def match_features(self, features: torch.Tensor) -> Dict[str, Union[List[int], int, str]]:
        """Match given features tensor with features tensor from database.

        Args:
            features: Tensor of features.

        Return:
            Dict of person info.
        """
        min_dist: float = 1000000.0
        idx: int = -1

        try:
            self._session.begin()
        except InvalidRequestError:
            self._session.close()

        query: Query = self._session.query(Face)
        
        for t in query:
            dist: float = distance(features, t.tensor)
            if dist < min_dist:
                min_dist = dist
                idx: int = t.person_id

        self._session.close()

        if min_dist <= self.max_distance:
            id_: int = idx
        else:
            id_: int = None
        
        data: Dict[str, Union[List[int], Optional[int]]] = {'bbox': [],
                                                            'id': id_}
        return data


def distance(features1: torch.Tensor, features2: torch.Tensor) -> float:
    """Calculate distance between given features tensors.

    Args:
        features1: First tensor of features.
        features2: Second tensor of features.

    Return:
        Calculated distance.
    """
    dist: float = torch.norm(features1 - features2, dim=0)
    return dist
