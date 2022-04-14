#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

install_requires = [
    'numpy',
    'pandas',
    'networkx',
    'scikit-learn',
    'tensorflow==2.8.0',
    'matplotlib',
    'lightgbm',
    'tqdm',
    'dill',
    'pynverse',
    'scipy'
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
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords=['DATGAN', 'GAN', 'Synthetic Tabular Data', 'Population Synthesis'],
    name='datgan',
    packages=find_packages(include=['datgan', 'datgan.*']),
    python_requires='>=3.7',
    url='https://github.com/glederrey/DATGAN',
    version='2.1.5',
    zip_safe=False,
)