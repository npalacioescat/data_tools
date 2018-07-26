# -*- coding: utf-8 -*-

from setuptools import setup,  find_packages

from data_tools import __version__, __author__

setup(name='data_tools',
      version=__version__,
      description="Data tools: a collection of Python functions and classes designed to make data scientists' life easier.",
      url='https://github.com/Nic-Nic/data_tools',
      author=__author__,
      author_email='nicolaspalacio91@gmail.com',
      license='GNU-GLPv3',
      package_dir = {'':'src'},
      packages=find_packages('src'),
      install_requires=['numpy',
                        'matplotlib',
                        'pandas',
                        'scipy',
                        'sklearn'],
      zip_safe=False)
