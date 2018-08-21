# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='data_tools',
      version=u'0.0.6',
      description="Data tools: a collection of Python functions and classes designed to make data scientists' life easier.",
      url='https://github.com/Nic-Nic/data_tools',
      author=u'Nicolàs Palacio',
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
