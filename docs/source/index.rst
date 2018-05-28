.. data_tools documentation master file, created by
   sphinx-quickstart on Wed May  9 12:40:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################
data_tools's reference
######################

Collection of Python functions and classes designed to make a
Computational Biologist's life easier.


Copyright (C) 2018 Nicolàs Palacio

Contact: nicolaspalacio91@gmail.com

GNU-GLPv3:
This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

A full copy of the GNU General Public License can be found on file
`LICENSE.md <../../../LICENSE.md>`_. If not, see
http://www.gnu.org/licenses/.

Dependencies
============

In module **data_tools.plots**:

- `NumPy <http://www.numpy.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `Pandas <https://pandas.pydata.org/>`_

In module **data_tools.Lasso**:

- `NumPy <http://www.numpy.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `Pandas <https://pandas.pydata.org/>`_
- `Scikit-learn <http://scikit-learn.org/stable/index.html>`_


Installation
============

First download/clone ``data_tools`` from the
`GitHub repository <https://github.com/Nic-Nic/data_tools.git>`_.
From the terminal:

.. code-block:: bash

   git clone https://github.com/Nic-Nic/data_tools.git
   cd data_tools

Then you can install it by running ``setup.py`` as follows:

.. code-block:: bash

   python setup.py sdist

Or using ``pip``:

.. code-block:: bash

   pip install .

Reference
=========

.. toctree::
   :maxdepth: 5

   Lasso
   plots
   sets
   strings
