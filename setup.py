# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = ''

setup(
    long_description=readme,
    name='personalities',
    version='0.1.0',
    description='A library of AI learners.',
    python_requires='==3.*,>=3.6.0',
    author='SimLeek',
    author_email='simulator.leek@gmail.com',
    license='MIT',
    packages=['personalities', 'personalities.base', 'personalities.util'],
    package_dir={"": "."},
    package_data={"personalities.util": ["*.torch"]},
    install_requires=[],
    extras_require={"dev": ["pytest==5.*,>=5.2.0", "tox==3.*,>=3.14.0"]},
)
