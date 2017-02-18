#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
from setuptools import setup, find_packages, Extension


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

ext_modules = [
    Extension("elektronn2.malis._malis",
              sources=["elektronn2/malis/_malis.pyx", "elektronn2/malis/_malis_lib.cpp"],
              include_dirs=['malis/'],
              language='c++'),
]

ext_modules = []  # Temporary workaround for broken malis build

setup(
    name="elektronn2",
    version="0.1",
    packages=find_packages(),
    scripts=[
        'scripts/elektronn2-train',
    ],
    ext_modules=ext_modules,
    setup_requires=['cython>=0.23'],
    install_requires=[
        'cython>=0.23',
        'numpy>=1.8',
        'scipy>=0.14',
        'matplotlib>=1.4',
        'h5py>=2.2',
        'theano>=0.7',
        'future>=0.15',
        'tqdm>=4.5',
        'colorlog>=2.7',
        'prompt_toolkit>=1.0.3',
        'jedi>=0.9.0',
        'pydotplus',
        'seaborn',
        'scikit-learn<0.20',  # >=0.20 requires change, see todo-note in data.traindata import
        'psutil>=5.0.1',
        'scikit-image>=0.12.3',
        'numba>=0.25'
    ],
    include_package_data=True,
    author="Marius Killinger",
    author_email="Marius.Killinger@mailbox.org",
    description="ELEKTRONN2 a is highly configurable toolkit for training 3d/2d CNNs and general Neural Networks",
    long_description=read('README.rst'),
    license="GPL",
    keywords="cnn theano neural network machine learning classification",
    url="http://www.elektronn.org/",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Scientific/Engineering :: Information Analysis", ],
    # package_data={'': ['*.so']},
)
