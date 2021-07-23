from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from PIL.Image import Image
from typing import List, Tuple
from FRMS.database import Session, Face, PersonInfo, AvgFace
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


class FeatureExtractor:
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self._device)

    def extract_features(self, face: torch.Tensor) -> torch.Tensor:
        face = face.to(self._device)
        features = self._resnet(face.unsqueeze(0)).detach().cpu()
        return features[0]


class FeatureMatcher:
    def __init__(self, max_distance: float = 0.03, deviation_percent: float = 0.05):
        self.max_distance = max_distance
        self.deviation_percent = deviation_percent
        self._session = Session()
        self._avg_face = self._get_avg_face()
        self._query = self._session.query(Face, PersonInfo).join(PersonInfo, Face.person_id == PersonInfo.id)

    def match_features(self, features: torch.Tensor):
        dists, ids = [], []
        avg_dist = distance(features, self._avg_face)
        percent = avg_dist * self.deviation_percent
        query = self._query.filter(Face.distance < avg_dist + percent, Face.distance >= avg_dist - percent)
        for t, _ in query:
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

    def _get_avg_face(self) -> torch.Tensor:
        avg_face = self._session.query(AvgFace).one()
        return avg_face.tensor


def distance(features1: torch.Tensor, features2: torch.Tensor) -> float:
    dist = (torch.sqrt_((features1 - features2) ** 2).mean()).item()
    return dist
