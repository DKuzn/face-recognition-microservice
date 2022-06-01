from setuptools import setup
from FRMS import __version__

setup(
    name='FRMS',
    version=__version__,
    packages=['FRMS', 'FRMS.utils'],
    url='https://github.com/DKuzn/face-recognition-microservice',
    license='GPLv3',
    author='Dmitry Kuznetsov',
    author_email='DKuznetsov2000@outlook.com',
    description='Microservice for face recognition',
    install_requires=['fastapi', 'torch', 'torchvision', 'facenet_pytorch', 'SQLAlchemy', 'psycopg2-binary', 'mysqlclient', 'uvicorn']
)
