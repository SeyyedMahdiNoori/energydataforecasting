#!/usr/bin/env python3

from setuptools import setup

with open("README.md" , "r")as fh:
    long_description = fh.read()

setup(
    name='converge-load-forecasting',
    version='0.0.5',
    author='Seyyed Mahdi Noori Rahim Abadi, Dan Gordon',
    author_email='mahdi.noori@anu.edu.au',
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    packages=['converge_load_forecasting'],
    install_requires=[
        'connectorx',
        # 'dateutil==2.8.2',
        'datetime',
        'matplotlib',
        'more_itertools',
        'multiprocess',
        'pandas',
        'pyomo',
        'skforecast==0.4.2',
        'sklearn',
        'statsmodels',
        'tqdm',
        # 'tsprial'
    ],
    scripts=[
    ]
)
