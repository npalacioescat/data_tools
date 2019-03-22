.. data_tools documentation master file, created by
   sphinx-quickstart on Wed May  9 12:40:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################
data_tools' reference
######################

.. only:: html

   .. image:: https://img.shields.io/badge/release-v0.0.6-blue.svg
      :alt: GitHub release
   .. image:: https://img.shields.io/badge/tests-100%25_passed-brightgreen.svg
      :alt: Tests output

Data tools: a collection of Python functions and classes designed to
make data scientists' life easier.

Copyright (C) 2019 Nicolàs Palacio-Escat

Contact: `nicolas.palacio@bioquant.uni-heidelberg.de <mailto:nicolas.palacio@bioquant.uni-heidelberg.de>`_

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
`LICENSE.md <https://github.com/Nic-Nic/data_tools/blob/master/LICENSE.md>`_.
If not, see http://www.gnu.org/licenses/.

.. only:: html

   Contents
   ========

   - :py:mod:`databases <data_tools.databases>`
      - :py:func:`kegg_link <data_tools.databases.kegg_link>`
      - :py:func:`kegg_pathway_mapping <data_tools.databases.kegg_pathway_mapping>`
      - :py:func:`op_kinase_substrate <data_tools.databases.op_kinase_substrate>`
      - :py:func:`up_map <data_tools.databases.up_map>`

   - :py:mod:`diffusion <data_tools.diffusion>`
      - :py:func:`euler_explicit1D <data_tools.diffusion.euler_explicit1D>`
      - :py:func:`euler_explicit2D <data_tools.diffusion.euler_explicit2D>`
      - :py:func:`euler_implicit_coef_mat <data_tools.diffusion.euler_implicit_coef_mat>`
      - :py:func:`crank_nicolson_coef_mats <data_tools.diffusion.crank_nicolson_coef_mats>`
      - :py:func:`build_coef_mat <data_tools.diffusion.build_coef_mat>`

   - :py:mod:`iterables <data_tools.iterables>`
      - :py:func:`bit_or <data_tools.iterables.bit_or>`
      - :py:func:`chunk_this <data_tools.iterables.chunk_this>`
      - :py:func:`find_min <data_tools.iterables.find_min>`
      - :py:func:`in_all <data_tools.iterables.in_all>`
      - :py:func:`similarity <data_tools.iterables.similarity>`
      - :py:func:`subsets <data_tools.iterables.subsets>`
      - :py:func:`unzip_dicts <data_tools.iterables.unzip_dicts>`

   - :py:mod:`models <data_tools.models>`
      - :py:class:`DoseResponse <data_tools.models.DoseResponse>`

        - :py:meth:`ec <data_tools.models.DoseResponse.ec>`
        - :py:meth:`plot <data_tools.models.DoseResponse.plot>`

      - :py:class:`Lasso <data_tools.models.Lasso>`

        - :py:meth:`fit_data <data_tools.models.Lasso.fit_data>`
        - :py:meth:`plot_score <data_tools.models.Lasso.plot_score>`
        - :py:meth:`plot_coef <data_tools.models.Lasso.plot_coef>`

   - :py:mod:`plots <data_tools.plots>`
      - :py:data:`cmap_bkgr <data_tools.plots.cmap_bkgr>`
      - :py:data:`cmap_bkrd <data_tools.plots.cmap_bkrd>`
      - :py:data:`cmap_rdbkgr <data_tools.plots.cmap_rdbkgr>`
      - :py:func:`density <data_tools.plots.density>`
      - :py:func:`piano_consensus <data_tools.plots.piano_consensus>`
      - :py:func:`venn <data_tools.plots.venn>`
      - :py:func:`volcano <data_tools.plots.volcano>`

   - :py:mod:`spatial <data_tools.spatial>`
      - :py:func:`get_boundaries <data_tools.spatial.get_boundaries>`
      - :py:func:`neighbour_count <data_tools.spatial.neighbour_count>`

   - :py:mod:`strings <data_tools.strings>`
      - :py:func:`is_numeric <data_tools.strings.is_numeric>`
      - :py:func:`join_str_lists <data_tools.strings.join_str_lists>`

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
   iterables
   models
   plots
   spatial
   strings
