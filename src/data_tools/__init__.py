# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Data tools: a collection of Python functions and classes designed to
make data scientists' life easier.


Copyright (C) 2019 Nicolàs Palacio-Escat

Contact: nicolas.palacio@bioquant.uni-heidelberg.de

GNU-GLPv3:
This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

A full copy of the GNU General Public License can be found on file
"LICENSE.md". If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import

import data_tools.__info__ as __info__
import data_tools.databases as databases
import data_tools.diffusion as diffusion
import data_tools.iterables as iterables
import data_tools.models as models
import data_tools.plots as plots
import data_tools.signal as signal
import data_tools.spatial as spatial
import data_tools.strings as strings

__version__ = __info__.__version__
__author__ = __info__.__author__
