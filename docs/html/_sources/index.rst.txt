.. data_tools documentation master file, created by
   sphinx-quickstart on Wed May  9 12:40:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################
data_tools' reference
######################

.. only:: html

   .. image:: https://img.shields.io/badge/release-v0.0.5-blue.svg
      :alt: GitHub release
   .. image:: https://img.shields.io/badge/tests-100%25_passed-brightgreen.svg
      :alt: Tests output

Data tools: a collection of Python functions and classes designed to
make data scientists' life easier.

Copyright (C) 2018 Nicolàs Palacio

Contact: `nicolaspalacio91@gmail.com <mailto:nicolaspalacio91@gmail.com>`_

Disclaimer
==========

This package is still under development and will be periodically updated
with new features. Contributions are very welcome (fork + pull request).
If you find any bug or suggestion for upgrades, please use the
`issue system <https://github.com/Nic-Nic/data_tools/issues>`_.


GNU-GLPv3:
This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

A full copy of the GNU General Public License can be found on file
`LICENSE.md <../../LICENSE.md>`_. If not, see
http://www.gnu.org/licenses/.

.. only:: html

   Contents
   ========

   - :meth:`data_tools.databases`
      - :meth:`data_tools.databases.kegg_link`
      - :meth:`data_tools.databases.kegg_pathway_mapping`
      - :meth:`data_tools.databases.up_map`
   - :meth:`data_tools.diffusion`
      - :meth:`data_tools.diffusion.euler_explicit1D`
   - :meth:`data_tools.models`
      - :meth:`data_tools.models.Lasso`
   - :meth:`data_tools.plots`
      - :meth:`data_tools.plots.density`
      - :meth:`data_tools.plots.piano_consensus`
      - :meth:`data_tools.plots.venn`
      - :meth:`data_tools.plots.volcano`
   - :meth:`data_tools.sets`
      - :meth:`data_tools.sets.bit_or`
      - :meth:`data_tools.sets.find_min`
      - :meth:`data_tools.sets.in_all`
      - :meth:`data_tools.sets.subsets`
   - :meth:`data_tools.strings`
      - :meth:`data_tools.strings.is_numeric`
      - :meth:`data_tools.strings.join_str_lists`

Dependencies
============

- `NumPy <http://www.numpy.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `Pandas <https://pandas.pydata.org/>`_
- `SciPy <https://www.scipy.org/>`_
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

Along with ``data_tools``, all dependencies will be installed as well as
the testing suite. In order to run the tests, type on the terminal:

.. code-block:: bash

   python -m test_data_tools

**NOTE:** ``data_tools.plots`` module does not have any tests
implemented.

.. only:: html

   Documentation
   =============

   You can find a detailed description of the modules and functions
   within this package in this webpage, also available as a
   :download:`PDF <../latex/data_tools.pdf>`.

   You can also find the version history in the
   `changelog <https://github.com/Nic-Nic/data_tools/blob/master/CHANGELOG.md>`_.

Modules
=======

.. toctree::
   :maxdepth: 5

   databases
   diffusion
   models
   plots
   sets
   strings
