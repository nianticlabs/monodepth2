# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='monodepth2',
    version='1.0.0',
    description='Monocular depth estimation from a single image',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nianticlabs/monodepth2',
    author='Niantic, Inc.',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        # only required for training and opencv is a bit tricky to install
        # 'tensorboardX',
        # 'opencv-python',
    ],
)
