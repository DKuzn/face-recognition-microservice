from sqlalchemy import Column, Integer, String, Float, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import torch
import os


DATABASE_URL = os.environ['DATABASE_URL'].replace('postgres', 'postgresql')
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


Base = declarative_base()


class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True, autoincrement=True)
    features = Column(String)
    person_id = Column(Integer, ForeignKey('person_info.id'))

    def __init__(self, tensor: torch.Tensor, person_id: int):
        self.features = self._tensor_to_str(tensor)
        self.person_id = person_id

    @property
    def tensor(self):
        return torch.tensor(list(map(float, self.features.split(' '))))

    @staticmethod
    def _tensor_to_str(t):
        return ' '.join([str(i) for i in t.tolist()])

    def __repr__(self):
        return "<Face('%s','%s')>" % (self.features, self.person_id)


class PersonInfo(Base):
    __tablename__ = 'person_info'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    surname = Column(String)

    def __init__(self, name, surname):
        self.name = name
        self.surname = surname

    def __repr__(self):
        return "<PersonInfo('%s','%s')>" % (self.name, self.surname)


if __name__ == '__main__':
    Base.metadata.create_all(engine)
    users_table = Face.__tablename__, PersonInfo.__tablename__
    print(users_table)
    metadata = Base.metadata
    print(metadata)
