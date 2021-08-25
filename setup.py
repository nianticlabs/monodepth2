import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="monodepth2",
    version="0.1.0",
    packages=['.', 'networks'],
    scripts=[],
    license='LICENSE',
    url='https://github.com/AdityaNG/monodepth2',
    description="[ICCV 2019] Monocular depth estimation from a single image",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
)
