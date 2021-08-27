# FRMS/utils/feature_matcher.py
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
"""This module contains class FeatureMatcher and function distance.

Features matches by calculating distance between given features tensor
and features tensors from database.
"""

from FRMS.database import Session, Face
from sqlalchemy.orm import Query
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
        self._query: Query = self._session.query(Face)

    def match_features(self, features: torch.Tensor) -> Dict[str, Union[List[int], int, str]]:
        """Match given features tensor with features tensor from database.

        Args:
            features: Tensor of features.

        Return:
            Dict of person info.
        """
        dists: List[float] = []
        ids: List[int] = []
        for t in self._query:
            dists.append(distance(features, t.tensor))
            ids.append(t.person_id)

        min_index: Optional[int] = self._min_dist(dists)
        if min_index is not None:
            id_: int = ids[min_index]
        else:
            id_: None = None
        data: Dict[str, Union[List[int], Optional[int]]] = {'bbox': [],
                                                            'id': id_}
        return data

    def _min_dist(self, dists: List[float]) -> Optional[int]:
        """Find minimal distance and compare it with threshold.

        Args:
            dists: List of distances.

        Return:
            Index if distance is below threshold. None otherwise.
        """
        min_dist: float = 1
        index: Optional[int] = None
        for idx, i in enumerate(dists):
            if i < min_dist:
                min_dist = i
                index = idx
        if min_dist <= self.max_distance:
            return index
        else:
            return None


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
