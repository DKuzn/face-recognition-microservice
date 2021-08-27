# FRMS/database.py
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
"""This module contains classes to work with database.

Note:
    Set environment variable DATABASE_URL to correct work of this module.

Example:
    >>> from FRMS.database import Session, Face
    >>> session = Session()
    >>> for face in session.query(Face)
    ...     print(face)
"""

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker
import torch
import os


DATABASE_URL: str = os.environ['DATABASE_URL'].replace('postgres', 'postgresql')
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


Base: DeclarativeMeta = declarative_base()


class Face(Base):
    """Class to work with table 'faces'.

    Args:
        tensor: Features tensor.
        person_id: ID of the person.

    Attributes:
        id: Primary key of table (set automatically).
        features: Features string.
        person_id: ID of the person.
    """
    __tablename__: str = 'faces'
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    features: str = Column(String)
    person_id: int = Column(Integer)

    def __init__(self, tensor: torch.Tensor, person_id: int) -> None:
        self.features = self._tensor_to_str(tensor)
        self.person_id = person_id

    @property
    def tensor(self) -> torch.Tensor:
        """Convert string features to tensor."""
        return torch.tensor(list(map(float, self.features.split(' '))))

    @staticmethod
    def _tensor_to_str(tensor) -> str:
        """Convert tensor to string.

        Args:
            tensor: Features tensor

        Return:
            Tensor string.
        """
        return ' '.join([str(i) for i in tensor.tolist()])

    def __repr__(self) -> str:
        return "<Face('%s','%s')>" % (self.features, self.person_id)


if __name__ == '__main__':
    Base.metadata.create_all(engine)
    users_table = Face.__tablename__
    print(users_table)
    metadata = Base.metadata
    print(metadata)
