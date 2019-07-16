#!/usr/bin/env python3
from setuptools import setup
from setuptools import find_packages

setup(
    name="md3d",
    version="1.0",
    description="Object Detection Framework",
    packages=find_packages(),
    install_requires=['matplotlib',
                      'numpy>=1.12.0',
                      'opencv-python',
                      'pandas',
                      'pillow',
                      'protobuf',
                      'scipy',
                      'sklearn'
                      ])
