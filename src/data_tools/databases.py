# -*- coding: utf-8 -*-

'''
data_tools.databases
====================

Databases functions module.
'''

__all__ = ['up_map']

import urllib
import urllib2

import pandas as pd


def up_map(query, in_mode='ACC', out_mode='GENENAME'):
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

    * Example:
        >>> my_query = ['P00533', 'P31749', 'P16220']
        >>> up_map(my_query)
              ACC GENENAME
        0  P00533     EGFR
        1  P31749     AKT1
        2  P16220    CREB1
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

    df = to_df(page, header=True)
    df.columns = [in_mode, out_mode]

    return df


###############################################################################


def to_df(page, header=False):
    if header:
        aux = [i.split('\t') for i in page.split('\n')[:-1]]
        return pd.DataFrame(aux[1:], columns=aux[0])

    else:
        return pd.DataFrame([i.split('\t') for i in page.split('\n')[:-1]])
