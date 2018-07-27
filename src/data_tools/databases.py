# -*- coding: utf-8 -*-

'''
data_tools.databases
====================

Databases functions module.
'''

__all__ = ['kegg_link', 'up_map']

import urllib
import urllib2

import pandas as pd


def kegg_link(query, target='pathway'):
    '''
    Queries a request to the KEGG database to find related entries using
    cross-references. A list of available database(s) and query examples
    can be found in https://www.kegg.jp/kegg/rest/keggapi.html#link.

    * Arguments:
        - *query* [list]: Or any iterable type containing the
          identifier(s) to be queried as [str]. These can be either
          valid database identifiers or databases *per se* (see the link
          above).
        - *target* [str]: Optional, ``'pathway'`` by default. Targeted
          database to which the query should be linked to. You can check
          other options available in the URL above.

    * Returns:
        - [pandas.DataFrame]: Two-column table containing both the
          input query identifiers and their linked ones.

    * Example:
        >>> my_query = ['hsa:10458', 'ece:Z5100']
        >>> kegg_link(my_query)
           query        pathway
        0  hsa:10458  path:hsa04520
        1  hsa:10458  path:hsa04810
        2  ece:Z5100  path:ece05130
    '''

    url = 'http://rest.kegg.jp/link'

    data = '+'.join(query)
    request = urllib2.Request('/'.join([url, target, data]))

    response = urllib2.urlopen(request)
    page = response.read(200000)

    df = to_df(page, header=False)
    df.columns = ['query', target]

    return df


def up_map(query, source='ACC', target='GENENAME'):
    '''
    Queries a request to UniProt.org in order to map a given list of
    identifiers. You can check the options available of input/output
    identifiers at https://www.uniprot.org/help/api_idmapping.

    * Arguments:
        - *query* [list]: Or any iterable type containing the
          identifiers to be queried as [str].
        - *source* [str]: Optional, ``'ACC'`` by default. This is,
          UniProt accesion number. You can check other options available
          in the URL above.
        - *target* [str]: Optional, ``'GENENAME'`` by default. You can
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

    params = {'from':source,
              'to':target,
              'format':'tab',
              'query':' '.join(query)}

    data = urllib.urlencode(params)
    request = urllib2.Request(url, data)

    response = urllib2.urlopen(request)
    page = response.read(200000)

    df = to_df(page, header=True)
    df.columns = [source, target]

    return df


###############################################################################


def to_df(page, header=False):
    if header:
        aux = [i.split('\t') for i in page.split('\n')[:-1]]
        return pd.DataFrame(aux[1:], columns=aux[0])

    else:
        return pd.DataFrame([i.split('\t') for i in page.split('\n')[:-1]])
