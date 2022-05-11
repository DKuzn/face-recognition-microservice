# FRMS/dataset.py
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
"""This module contains class FaceDataset.

This class may be used for training face classification model and
adding faces to database (automatic face detection).
"""

from torch.utils.data import Dataset
from FRMS.utils.face_detector import FaceDetector
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import os


class FacesDataset(Dataset):
    """Dataset of faces.

    Args:
        root: Dataset root directory.

    Attributes:
        root: Dataset root directory.
        paths: List of image paths.
        labels_names: List of label names.
    """
    def __init__(self, root: str) -> None:
        super(FacesDataset, self).__init__()
        self.root: Path = Path(root)
        self.paths: List[str] = self._list_dirs()
        self.labels_names: List[str] = self._list_labels_names()
        self._label_indexes: Dict[str, int] = self._list_label_indexes()
        self._image_labels: List[int] = self._list_labels()
        self._face_detector: FaceDetector = FaceDetector()

    def __len__(self) -> int:
        """Get dataset length.

        Return:
            Dataset length.
        """
        return len(self.paths)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.

        Return two tensors: image and target label.

        Args:
            item: Index of item.

        Return:
            Tuple of tensors.
        """
        label: int = self._image_labels[item]
        img: Image.Image = Image.open(self.paths[item]).convert('RGB')
        x: torch.Tensor = self._face_detector.find_faces(img)[0][0]
        y: torch.Tensor = torch.tensor(label)
        return x, y

    def _list_dirs(self) -> List[str]:
        paths: List[Path] = list(self.root.rglob('*.*'))
        paths: List[str] = [str(path) for path in paths]
        return sorted(paths)

    def _list_labels_names(self) -> List[str]:
        labels: List[str] = sorted(item.name for item in self.root.glob('*/') if item.is_dir())
        return labels

    def _list_label_indexes(self) -> Dict[str, int]:
        label_to_index: Dict[str, int] = dict((name, index) for index, name in enumerate(self.labels_names))
        return label_to_index

    def _list_labels(self) -> List[int]:
        image_labels: List[int] = [self._label_indexes[Path(path).parent.name] for path in self.paths]
        return image_labels

    def get_face_id(self, item) -> int:
        """Get face ID from dataset.
        
        Args:
            item: Index of item.
            
        Return:
            Face ID.
        """
        face_id: str = str(Path(self.paths[item]).parent).split(os.sep)[-1]
        return int(face_id)


if __name__ == '__main__':
    ds = FacesDataset('../database')
    print(len(ds))
    print(ds.labels_names)
    for i in range(10):
        print(ds[i][1])
        print(ds.get_face_id(i))
