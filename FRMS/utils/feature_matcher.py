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
