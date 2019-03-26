# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import time
import unittest

import test_data_tools


if __name__ == '__main__':
    start = time.time()

    print '\nWelcome to data_tools test suite'
    print '################################\n'
    print 'Session started @ %s' % time.strftime('%a %d %b %Y - %H:%M')

    test_loader = unittest.TestLoader()
    run = []
    skip = []
    fail = []

    for submod in [m for m in dir(test_data_tools) if m.startswith('test_')]:
        msg = 'Running tests for submodule %s' % submod.replace('test_', '')
        print '\n%s\n%s' % (msg, '=' * len(msg))

        test_suite = test_loader.loadTestsFromModule(getattr(test_data_tools,
                                                             submod))
        res = unittest.TextTestRunner(verbosity=2).run(test_suite)
        run.append(res.testsRun)
        skip.append(len(res.skipped))
        fail.append(len(res.errors))

    if sum(fail) == 0:
        msg = 'OK'

    else:
        msg = 'FAILED (errors=%d)' % sum(fail)

    if sum(skip) != 0:
        msg += ' (skipped=%d)' % sum(skip)

    print '=' * 70
    print 'TOTAL ran %d tests in %.3fs\n' % (sum(run), time.time() - start)
    print msg
