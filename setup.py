# -*- coding: utf-8 -*-

import sys
import os
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

__info__ = SourceFileLoader(
        '__info__',
        os.path.join('src', 'data_tools', '__info__.py')
).load_module()

mpl = 'matplotlib < 3' if sys.version_info < (3,) else 'matplotlib',

setup(name='data_tools',
      version=__info__.__version__,
      description="Data tools: a collection of Python functions and classes"
                  "designed to make data scientists' life easier.",
      url='https://github.com/Nic-Nic/data_tools',
      author=__info__.__author__,
      author_email='nicolas.palacio@bioquant.uni-heidelberg.de',
      license='GNU-GLPv3',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=['numpy',
                        mpl,
                        'pandas',
                        'scipy',
                        'scikit-learn',
                        'UpSetPlot',
                        'adjustText',
                        'certifi'],
      zip_safe=False)
