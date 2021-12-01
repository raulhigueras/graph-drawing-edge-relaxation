# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import requests
import json
import pandas as pd
import networkx as nx
from src.graph_dataset import GraphDataset
from src.graph_parser import parseGtFile
from src.graph_utils import prettyPos, num_crossings


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Checking data from networks.skewed.de...')

    res_txt = requests.get('https://networks.skewed.de/api/nets?full=True').text
    test_json = json.loads(res_txt)

    cols = ['name', 'V', 'E', 'DIRECTED', 'TAGS']
    df = pd.DataFrame(columns=cols)
    ls = list(test_json.items())
    for g_name, d in ls:
        if len(d['nets']) > 100: pass
        elif len(d['nets']) == 1:
            a = d['analyses']
            data = dict(zip(cols, [g_name, a['num_vertices'], a['num_edges'], a['is_directed'], d['tags']]))
            df = df.append(data, ignore_index=True)
        else:
            for net in d['nets']:
                a = d['analyses'][net]
                name = f'{g_name}/{net}'
                data = dict(zip(cols, [name, a['num_vertices'], a['num_edges'], a['is_directed'], d['tags']]))
                df = df.append(data, ignore_index=True)

    logger.info('Filtering graphs to match restrictions...')

    #MIN_V, MAX_V = 50, 600
    MIN_E, MAX_E = 50, 1000
    ALLOW_DIRECTED = False
    EXCLUDED_TAGS = ['Temporal', 'Timestamps', 'Multigraph']

    filtered_df = df[(df['E'] >= MIN_E) & (df['E'] <= MAX_E)]
    if not ALLOW_DIRECTED: 
        filtered_df = filtered_df[filtered_df['DIRECTED'] == False]
    where_tags = [not any(x in row[4] for x in EXCLUDED_TAGS) for row in filtered_df.values]
    filtered_df = filtered_df[where_tags].reset_index(drop=True)
    n_graphs = filtered_df.shape[0]

    logger.info(f'# graphs after 1st filtering: {n_graphs}')
    logger.info(f'Downloading and parsing graph data + filter(this may take a while)...')

    gd = GraphDataset()

    bibtexts = []

    for i, (name, tags) in enumerate(zip(filtered_df["name"], filtered_df["TAGS"])):
        n = name.split("/")
        n1,n2 = n if len(n)==2 else (n[0],n[0]) 
        g = parseGtFile(f'https://networks.skewed.de/net/{n1}/files/{n2}.gt.zst', False, False)
        g.remove_edges_from(nx.selfloop_edges(g))                           # remove self-loops
        if not nx.is_connected(g):
            g = g.subgraph(max(nx.connected_components(g), key=len)).copy() # select one component
            g = nx.convert_node_labels_to_integers(g)
            if not (MIN_E <= len(g.edges) <= MAX_E):                        # filter out small graphs 
                continue
            ncross = num_crossings(g, prettyPos(g))
            if ncross == 0:                                                 # filter out graphs w 0 cross
                continue
        #gd.addGraph(g, name=name, type=tags)
        bibtexts.append(json.loads(requests.get(f'https://networks.skewed.de/api/net/{n1}').text)['bibtex'])
        if (i+1) % 10 == 0:
            logger.info(f'Parsed: {i+1}/{n_graphs}')

    #bibtexts = set(bibtexts)
    print(bibtexts)

    return 

    logger.info(f'graphs after second filtering: {len(gd)}')

    logger.info(f'Exporting dataset into: {output_filepath}')
    
    gd.export(output_filepath, 'skewed')

    logger.info(f'Dataset generated! :)')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
