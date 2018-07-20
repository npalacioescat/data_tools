# -*- coding: utf-8 -*-
#!/usr/bin/env python

import unittest

import test_data_tools


if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_data_tools.__path__[0])

    unittest.TextTestRunner(verbosity=2).run(test_suite)

    print 'NOTE: data_tools.plots module has not been tested'
