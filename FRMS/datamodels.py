# FRMS/datamodels.py
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
"""This module contains data validation models.

Data validation provided bu Pydantic.
"""

from pydantic import BaseModel
from typing import List, Optional


class RequestModel(BaseModel):
    """Request format to microservice.

    Attributes:
        image: Base64 image.
    """
    image: str


class ResponseModel(BaseModel):
    """Response format of microservice.

    Attributes:
        bbox: Coordinates of bounding box.
        id: ID of person.
        name: Name of person.
        surname: Surname of person.
    """
    bbox: List[int]
    id: Optional[int]
    name: str
    surname: str
