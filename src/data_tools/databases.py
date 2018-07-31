# -*- coding: utf-8 -*-

'''
data_tools.databases
====================

Databases functions module.
'''

__all__ = ['kegg_link', 'kegg_pathway_mapping', 'up_map']

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

    if df.iloc[0, 0] != '':
        df.columns = ['query', target]

    return df.sort_values(by=df.columns[0])


def kegg_pathway_mapping(df, mapid, filename=None):
    '''
    Makes a request to KEGG pathway mapping tool according to a given
    pathway ID (see https://www.kegg.jp/kegg/tool/map_pathway2.html for
    more information). The user must provide a query of IDs to be mapped
    with their corresponding background colors (and optionally also
    foreground colors). The result is downloaded in the current
    directory or a user-specified path.

    * Arguments:
        - *df* [pandas.DataFrame]: Dataframe containing KEGG valid IDs
          in the first column and corresponding background colors (e.g.:
          red, blue, ...). Optionally, a third column with the
          foreground (font) colors can also be provided (black by
          default). **NOTE:** hexadecimal codes for colors is also
          supported. Index and column names of dataframe are ignored.
        - *mapid* [str]: A valid KEGG pathway ID. It can be a general
          (e.g.: "mapXXXXX") or organism-specific ID (e.g.: "hsaXXXXX").
        - *filename* [str]: Optional, ``None`` by default. This is, the
          image will be stored in the current directory with the *mapid*
          provided as file name. If provided, the image will be stored
          within the specified path/file name.

    * Example:
        >>> my_query = pandas.DataFrame([['1956', 'red', '#f1f1f1'],
        ...                              ['3845', 'blue', '#f1f1f1'],
        ...                              ['5594', 'green', 'black']])
        >>> kegg_pathway_mapping(my_query, 'hsa04010')

        .. image:: ../figures/hsa04010.png
           :align: center
    '''

    url = 'https://www.kegg.jp'

    # If fgcolor is not provided set black as default
    if df.shape[1] == 2:
        df['fgcolor'] = ['black'] * len(df)

    query = '%0d%0a'.join(['%s+%s,%s' %(dbentry,
                                        bgc.replace('#', '%23'),
                                        fgc.replace('#', '%23'))
                           for i, (dbentry, bgc, fgc) in df.iterrows()])

    params = '/kegg-bin/show_pathway?map=%s&multi_query=%s' %(mapid, query)

    if mapid.endswith('01100'):
        return ('Skipping the query for %s: Metabolic Pathways.\nToo ' %mapid +
                'much abstraction to show any relevant information.\nYou ' +
                'can explore your query here:\n' + url + params)

    request = urllib2.Request(url + params)

    response = urllib2.urlopen(request)
    page = response.read(200000)

    # Now extract the image from the HTML page
    # There must be a cleaner way to parse the HTML file, but...
    end = page.find('" name="pathwayimage"')
    start = 10 + page.find('<img src="')

    params = page[start:end]

    if not filename:
        filename = '%s.png' %mapid

    urllib.urlretrieve(url + params, filename)


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

    * Examples:
        >>> my_query = ['P00533', 'P31749', 'P16220']
        >>> up_map(my_query)
              ACC GENENAME
        0  P00533     EGFR
        1  P31749     AKT1
        2  P16220    CREB1
        >>> up_map(my_query, target='KEGG_ID')
              ACC   KEGG_ID
        0  P00533  hsa:1956
        2  P16220  hsa:1385
        1  P31749   hsa:207
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

    return df.sort_values(by=df.columns[0])


###############################################################################


def to_df(page, header=False):
    if header:
        aux = [i.split('\t') for i in page.split('\n')[:-1]]
        return pd.DataFrame(aux[1:], columns=aux[0])

    else:
        return pd.DataFrame([i.split('\t') for i in page.split('\n')[:-1]])
