from setuptools import find_namespace_packages, setup

setup(
    name='chunkoptim',
    version='2.0',
    packages=find_namespace_packages(include=["chunkoptim", "chunkoptim.*"]),
    install_requires=[]
)
