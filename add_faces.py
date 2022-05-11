# add_faces.py
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
"""
This file contains the script to fill the database of a face features.

Note:
    Set the environment variable DATABASE_URL to correct work of this script.

Example:
    >>> python add_faces.py path/to/dataset
"""

from FRMS.database import Session, Face, create_table
from FRMS.dataset import FacesDataset
from FRMS.utils.feature_extractor import FeatureExtractor
from torch import Tensor
from tqdm import tqdm
import argparse
import os


def fill_database(path_to_dataset: str) -> None:
    """Fills the database of a face features.

    Args:
        path_to_dataset: The path to a dataset root directory.

    Return:
        None
    """
    ds: FacesDataset = FacesDataset(path_to_dataset)
    session: Session = Session()
    feature_extractor: FeatureExtractor = FeatureExtractor()
    faces_count: int = len(ds)

    for i in tqdm(range(faces_count)):

        features: Tensor = feature_extractor.extract_features(ds[i][0])
        session.add(Face(features, ds.get_face_id(i)))

    session.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to fill the database of a face features.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('path', metavar='path/to/dataset', type=str, 
                        help="""The path to the dataset with a faces images.
Dataset format:
|--root/
    |--1
        |--somename.jpg
    |--2
        |--somename.jpg
    ...................
    |--N
        |--somename.jpg""")
    args = parser.parse_args()
    try:
        if not os.path.exists(args.path):
            raise FileNotFoundError
        create_table()
        fill_database(args.path)
    except FileNotFoundError:
        print('The path to the dataset is incorrect.')
    