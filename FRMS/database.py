# FRMS/database.py
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
"""This module contains classes to work with database.

Note:
    Set environment variable DATABASE_URL to correct work of this module.

Example:
    >>> from FRMS.database import Session, Face
    >>> session = Session()
    >>> for face in session.query(Face)
    ...     print(face)
"""

from sqlalchemy import Column, Integer, PickleType, create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker
from typing import List
import torch
import os


DATABASE_URL: str = os.environ['DATABASE_URL'].replace('postgres://', 'postgresql://')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)


Base: DeclarativeMeta = declarative_base()


class Face(Base):
    """Class to work with table 'faces'.

    Args:
        tensor: Features tensor.
        person_id: ID of the person.

    Attributes:
        id: Primary key of table (set automatically).
        features: Features list.
        person_id: ID of the person.
    """
    __tablename__: str = 'faces'
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    features: List[float] = Column(PickleType)
    person_id: int = Column(Integer)

    def __init__(self, tensor: torch.Tensor, person_id: int) -> None:
        self.features = tensor.tolist()
        self.person_id = person_id

    @property
    def tensor(self) -> torch.Tensor:
        """Convert list features to tensor."""
        return torch.tensor(self.features)

    def __repr__(self) -> str:
        return "<Face('%s','%s')>" % (self.features, self.person_id)


def create_table() -> None:
    """Creates the table in the database.

    Return:
        None
    """
    Base.metadata.create_all(engine)
    table: str = Face.__tablename__
    print(table)


if __name__ == '__main__':
    create_table()
