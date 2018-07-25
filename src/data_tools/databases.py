# -*- coding: utf-8 -*-

'''
data_tools.databases
====================

Databases functions module.
'''

__all__ = []

import urllib
import urllib2
import pandas as pd


def up_query(query, in_mode='ACC', out_mode='GENENAME'):
    '''
    Queries a request to UniProt.org in order to map a list of
    identifiers
    '''

    url = 'https://www.uniprot.org/uploadlists/'

    params = {'from':in_mode,
              'to':out_mode,
              'format':'tab',
              'query':' '.join(query)}

    data = urllib.urlencode(params)
    request = urllib2.Request(url, data)

    response = urllib2.urlopen(request)
    page = response.read(200000)

    return [i.split('\t')[1] for i in page.split('\n')[1:-1]]
