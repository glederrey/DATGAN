#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'numpy>=1.16.0',
    'pandas>=1.3',
    'networkx>=2.6',
    'scikit-learn>=1.0.2',
    'tensorflow>=2.8.0',
    'matplotlib>=3.5',
    'lightgbm>=3.3',
    'tqdm>=4.62.3'
]

setup(
    author="Gael Lederrey",
    author_email='gael.lederrey@epfl.ch',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    description="Generative adversarial network with integrated expert knowledge for synthesizing tabular data",
    entry_points={
        'console_scripts': [
            'datgan=datgan.cli:main'
        ]
    },
    install_package_data=True,
    install_requires=install_requires,
    license="GPLv3",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='datgan',
    name='datgan',
    packages=find_packages(include=['datgan', 'datgan.*']),
    python_requires='>=3.9',
    url='https://github.com/glederrey/DATGAN',
    version='2.0.0',
    zip_safe=False,
)