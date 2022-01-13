# add_faces.py
#
# Copyright (c) 2021 Дмитрий Кузнецов
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
    