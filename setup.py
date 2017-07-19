#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
from setuptools import setup, find_packages, Extension


malis = False  # Set to True to enable malis build.

ext_modules = []
if malis:
    print('Trying to build the malis extension.\n'
          'Requires a C++ compiler, Numpy, Cython and Boost to be installed.')
    import numpy as np

    ext_modules = [
        Extension(
            "elektronn2.malis._malis",  # Note: _malis_lib.cpp requires boost!
            sources=[
                "elektronn2/malis/_malis.pyx",
                "elektronn2/malis/_malis_lib.cpp",
            ],
            include_dirs=[
                'malis/',
                np.get_include(),
            ],
            language='c++'
        ),
    ]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="elektronn2",
    version="0.3.0",
    packages=find_packages(),
    scripts=[
        'scripts/elektronn2-train',
        'scripts/elektronn2-profile',
    ],
    ext_modules=ext_modules,
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.14',
        'matplotlib>=1.4',
        'h5py>=2.2',
        'theano>=0.8,<0.10',  # ELEKTRONN2 relies on the old Theano backend that has been removed now
        'future>=0.15',
        'tqdm>=4.5',
        'colorlog>=2.7',
        'prompt_toolkit>=1.0.3',
        'jedi>=0.9.0',
        'pydotplus',
        'seaborn',
        'scikit-learn',
        'psutil>=5.0.1',
        'scikit-image>=0.12.3',
        'numba>=0.25',
    ],
    extras_require={
        'knossos': ['knossos_utils'],
        'ipython': ['ipython'],
    },
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis", ],
)
