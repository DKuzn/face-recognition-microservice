# FRMS/database.py
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
