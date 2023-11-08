#!/usr/bin/env python3

from setuptools import setup

with open("README.md" , "r")as fh:
    long_description = fh.read()

setup(
    name='energydataforecasting',
    version='v0.0.9',
    author='Seyyed Mahdi Noori Rahim Abadi, Dan Gordon',
    author_email='mahdi.noori@anu.edu.au',
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    packages=['energydataforecasting'],
    install_requires=[
        'python-dateutil',
        'datetime',
        'mapie',
        'matplotlib',
        'more_itertools',
        'multiprocess',
        'pandas',
        'pyomo',
        'pytz',
        'skforecast==0.7.0',
        'scikit-learn',
        'statsmodels',
        'threadpoolctl',
        'tqdm',
        'tspiral',
        'xgboost'
    ],
    scripts=[
    ]
)
