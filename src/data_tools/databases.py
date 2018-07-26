# -*- coding: utf-8 -*-

'''
data_tools.databases
====================

Databases functions module.
'''

__all__ = ['up_query']

import urllib
import urllib2

import pandas as pd


def up_query(query, in_mode='ACC', out_mode='GENENAME'):
    '''
    Queries a request to UniProt.org in order to map a given list of
    identifiers. You can check the options available of input/output
    identifiers at https://www.uniprot.org/help/api_idmapping.

    * Arguments:
        - *query* [list]: Or any iterable type containing the
          identifiers to be queried as [str].
        - *in_mode* [str]: Optional, ``'ACC'`` by default. This is,
          UniProt accesion number. You can check other options available
          in the URL above.
        - *out_mode* [str]: Optional, ``'GENENAME'`` by default. You can
          check other options available in the URL above.
    * Returns:
        - [pandas.DataFrame]: Two-column table containing both the
          inputed identifiers and the mapping result of these.
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

    result = [i.split('\t') for i in page.split('\n')[1:-1]]

    return pd.DataFrame(result, columns=[in_mode, out_mode])
