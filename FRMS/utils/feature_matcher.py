"""
FRMS/utils/feature_matcher.py

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

from FRMS.database import Session, Face, PersonInfo
import torch


class FeatureMatcher:
    def __init__(self, max_distance: float = 0.03):
        self.max_distance = max_distance
        self._session = Session()
        self._query = self._session.query(Face, PersonInfo).join(PersonInfo, Face.person_id == PersonInfo.id)

    def match_features(self, features: torch.Tensor):
        dists, ids = [], []
        for t, _ in self._query:
            dists.append(distance(features, t.tensor))
            ids.append(t.person_id)

        min_index = self._min_dist(dists)
        if min_index is not None:
            _, q = self._query.filter(PersonInfo.id == ids[min_index]).one()
            id_, name, surname = q.id, q.name, q.surname
        else:
            id_, name, surname = None, '-', '-'
        data = {'bbox': [],
                'id': id_,
                'name': name,
                'surname': surname}
        return data

    def _min_dist(self, dists: list):
        min_dist = 1
        index = None
        for idx, i in enumerate(dists):
            if i < min_dist:
                min_dist = i
                index = idx
        if min_dist <= self.max_distance:
            return index
        else:
            return None


def distance(features1: torch.Tensor, features2: torch.Tensor) -> float:
    dist = (torch.sqrt_((features1 - features2) ** 2).mean()).item()
    return dist
