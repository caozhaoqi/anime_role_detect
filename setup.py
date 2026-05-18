#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='ardc',
    version='1.0.0',
    description='Anime Role Detect - Skill Repository System',
    author='ARD Team',
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.100.0',
        'uvicorn>=0.23.2',
        'pydantic>=2.0.0',
        'prometheus-client>=0.17.1',
        'requests>=2.31.0',
        'python-multipart>=0.0.6'
    ],
    entry_points={
        'console_scripts': [
            'ardc=ardc.cli.cli:main',
        ],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)