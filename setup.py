#!/usr/bin/env python3

from setuptools import setup

setup(
    name='converge-load-forecasting',
    version='v0.0.1',
    author='Seyyed Mahdi Noori Rahim Abadi, Dan Gordon',
    author_email='mahdi.noori@anu.edu.au',
    packages=['converge_load_forecasting'],
    install_requires=[
        'matplotlib',
        'more_itertools',
        'multiprocess',
        'pandas',
        'pyomo',
        'skforecast==0.4.2',
        'sklearn',
        'statsmodels'
    ],
    scripts=[
    ]
)
