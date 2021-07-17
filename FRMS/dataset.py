from torch.utils.data import Dataset
from FRMS.utils import FeatureExtractor, FaceDetector
import torch
import pathlib
from PIL import Image


class FacesDataset(Dataset):
    def __init__(self, root):
        super(FacesDataset, self).__init__()
        self.root = pathlib.Path(root)
        self.paths = self._list_dirs()
        self.labels_names = self._list_labels_names()
        self._label_indexes = self._label_indexes()
        self._image_labels = self._labels()
        self._feature_extractor = FeatureExtractor()
        self._face_detector = FaceDetector()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        label = self._image_labels[item]
        img = Image.open(self.paths[item]).convert('RGB')
        x = self._face_detector.find_faces(img)[0]
        y = torch.tensor(label)
        return x, y

    def _list_dirs(self):
        paths = list(self.root.rglob('*.*'))
        paths = [str(path) for path in paths]
        return sorted(paths)

    def _list_labels_names(self):
        labels = sorted(item.name for item in self.root.glob('*/') if item.is_dir())
        return labels

    def _label_indexes(self):
        label_to_index = dict((name, index) for index, name in enumerate(self.labels_names))
        return label_to_index

    def _labels(self):
        image_labels = [self._label_indexes[pathlib.Path(path).parent.name] for path in self.paths]
        return image_labels

    def avg_face(self):
        avg_face = torch.zeros(512).detach()
        for t in range(len(self)):
            avg_face += self._feature_extractor.extract_features(self[t][0])
        return avg_face / len(self)

    def get_face_id(self, item):
        face_id = str(pathlib.Path(self.paths[item]).parent).split('/')[-1]
        return int(face_id)


if __name__ == '__main__':
    ds = FacesDataset('../database')
    print(len(ds))
    print(ds.labels_names)
    print(ds.avg_face())
    for i in range(10):
        print(ds[i][1])
        print(ds.get_face_id(i))
