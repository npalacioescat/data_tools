# data_tools

![GitHub release](https://img.shields.io/badge/release-v0.0.6-blue.svg)
![Tests output](https://img.shields.io/badge/tests-100%25_passed-brightgreen.svg)

Data tools: a collection of Python functions and classes designed to
make data scientists' life easier.

Copyright (C) 2019 Nicolàs Palacio-Escat

Contact: [nicolas.palacio@bioquant.uni-heidelberg.de](mailto:nicolas.palacio@bioquant.uni-heidelberg.de)

## Disclaimer

This package is still under development and will be periodically updated
with new features. Contributions are very welcome (fork + pull request).
If you find any bug or suggestion for upgrades, please use the [issue
system](https://github.com/Nic-Nic/data_tools/issues).


GNU-GLPv3:
This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

A full copy of the GNU General Public License can be found on file
[LICENSE.md](LICENSE.md). If not, see <http://www.gnu.org/licenses/>.

## Contents

- [databases](https://nic-nic.github.io/data_tools/html/databases.html)
    - [kegg_link](https://nic-nic.github.io/data_tools/html/databases.html#data_tools.databases.kegg_link)
    - [kegg_pathway_mapping](https://nic-nic.github.io/data_tools/html/databases.html#data_tools.databases.kegg_pathway_mapping)
    - [op_kinase_substrate](https://nic-nic.github.io/data_tools/html/databases.html#data_tools.databases.op_kinase_substrate)
    - [up_map](https://nic-nic.github.io/data_tools/html/databases.html#data_tools.databases.up_map)
- [diffusion](https://nic-nic.github.io/data_tools/html/diffusion.html)
    - [build_mat](https://nic-nic.github.io/data_tools/html/models.html#data_tools.diffusion.build_mat)
- [iterables](https://nic-nic.github.io/data_tools/html/iterables.html)
    - [bit_or](https://nic-nic.github.io/data_tools/html/iterables.html#data_tools.iterables.bit_or)
    - [chunk_this](https://nic-nic.github.io/data_tools/html/iterables.html#data_tools.iterables.chunk_this)
    - [find_min](https://nic-nic.github.io/data_tools/html/iterables.html#data_tools.iterables.find_min)
    - [in_all](https://nic-nic.github.io/data_tools/html/iterables.html#data_tools.iterables.in_all)
    - [similarity](https://nic-nic.github.io/data_tools/html/iterables.html#data_tools.iterables.similarity)
    - [subsets](https://nic-nic.github.io/data_tools/html/iterables.html#data_tools.iterables.subsets)
    - [unzip_dicts](https://nic-nic.github.io/data_tools/html/iterables.html#data_tools.iterables.unzip_dicts)
- [models](https://nic-nic.github.io/data_tools/html/models.html)
    - [DoseResponse](https://nic-nic.github.io/data_tools/html/models.html#data_tools.models.DoseResponse)
    - [Lasso](https://nic-nic.github.io/data_tools/html/models.html#data_tools.models.Lasso)
- [plots](https://nic-nic.github.io/data_tools/html/plots.html)
    - [cmap_bkgr](https://nic-nic.github.io/data_tools/html/plots.html#data_tools.plots.cmap_bkgr)
    - [cmap_bkrd](https://nic-nic.github.io/data_tools/html/plots.html#data_tools.plots.cmap_bkrd)
    - [cmap_rdbkgr](https://nic-nic.github.io/data_tools/html/plots.html#data_tools.plots.cmap_rdbkgr)
    - [density](https://nic-nic.github.io/data_tools/html/plots.html#data_tools.plots.density)
    - [piano_consensus](https://nic-nic.github.io/data_tools/html/plots.html#data_tools.plots.piano_consensus)
    - [venn](https://nic-nic.github.io/data_tools/html/plots.html#data_tools.plots.venn)
    - [volcano](https://nic-nic.github.io/data_tools/html/plots.html#data_tools.plots.volcano)
- [spatial](https://nic-nic.github.io/data_tools/html/spatial.html)
    - [get_boundaries](https://nic-nic.github.io/data_tools/html/spatial.html#data_tools.spatial.get_boundaries)
    - [neighbour_count](https://nic-nic.github.io/data_tools/html/spatial.html#data_tools.spatial.neighbour_count)
- [strings](https://nic-nic.github.io/data_tools/html/strings.html)
    - [is_numeric](https://nic-nic.github.io/data_tools/html/strings.html#data_tools.strings.is_numeric)
    - [join_str_lists](https://nic-nic.github.io/data_tools/html/strings.html#data_tools.strings.join_str_lists)

## Dependencies

- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit-learn](http://scikit-learn.org/stable/index.html)

## Installation

### Fast way with pip

If you have `pip`, just type on the terminal:

```bash
pip install git+https://github.com/Nic-Nic/data_tools.git
```

### By cloning the git repository

First download/clone `data_tools` from the [GitHub repository](https://github.com/Nic-Nic/data_tools.git).
From the terminal:

```bash
git clone https://github.com/Nic-Nic/data_tools.git
cd data_tools
```

Then you can install it by running `setup.py` as follows:

```bash
python setup.py sdist
```

Or using `pip`:

```bash
pip install .
```

### Testing the module

Along with `data_tools`, all dependencies will be installed as well as
the testing suite. In order to run the tests, type on the terminal:

```bash
python -m test_data_tools
```

**NOTE:** `data_tools.plots` module does not have any tests implemented.

## Documentation

You can find a detailed description of the modules and functions within
this package in the [webpage](https://nic-nic.github.io/data_tools/),
also available as a [PDF](/docs/latex/data_tools.pdf).

You can also find the version history in the [changelog](CHANGELOG.md).
