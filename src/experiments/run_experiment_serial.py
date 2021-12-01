# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import json
import pandas as pd
import networkx as nx
import numpy as np
import os
import itertools
from src.graph_dataset import GraphDataset
from src.graph_parser import parseGtFile
from src.graph_utils import prettyPos, num_crossings


def getLayout(g, k_r, k_w, max_it, patience):
    init_pos = prettyPos(g)
    init_cross = num_crossings(g, init_pos)

    scale = {e:1 for e in g.edges}
    weight = {e:1 for e in g.edges}
    eb = nx.centrality.edge_betweenness(g)

    best_cross = np.infty
    best_cross_it = -1
    last_pos = init_pos.copy()

    for it in range(max_it):
        values = {e:weight[e]*eb[e] for e in g.edges}
        amax = max(values.items(), key=lambda x: x[1])[0]
        scale[amax] *= k_r
        weight[amax] *= k_w
        nx.set_edge_attributes(g, scale, name='relax')

        pos = nx.spring_layout(g, pos=last_pos.copy(), weight='relax')
        
        ncross = num_crossings(g, pos)
        if ncross < best_cross:
            best_cross = ncross
            best_cross_it = it

        if best_cross_it + patience < it:
            break

        last_pos = pos
    
    return init_cross, best_cross, best_cross_it




@click.command()
@click.argument('experiment_num')
@click.argument('output_filepath', type=click.Path())
def main(experiment_num, output_filepath):
    """ Runs experiment using params{experiment_number}.json
        and saves the results in /data/experiment_results.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Loading parameters from params{experiment_num}.json')

    params = {}
    params_filename = f'{project_dir}/src/Experiments/params{experiment_num}.json'
    if not os.path.isfile(params_filename):
        logger.error(f"File params{experiment_num}.json does not exist!")
        return

    f = open(params_filename, 'r')
    params = json.loads(f.read())
    f.close()

    logger.info(f'Starting experiment with: {params}')

    ds = GraphDataset.fromFile(params['dataset_src']) # Dataset  
    idxs = np.argsort(ds.df_graphs['edges'])[::-1]

    col_names = ["graph_id", "k_relax", "k_weight", "patience", "max_it", "init_cross", "final_cross", "n_it"]
    res = pd.DataFrame( columns=col_names )

    hyperparams = itertools.product(params['k_relax'], params['k_weight'])

    for k_r, k_w in list(hyperparams):
        logger.info(f'Testing: {(k_r, k_w)}')
        for idx in idxs:
            g = ds.getGraph(idx).copy()

            init_cross, best_cross, best_cross_it = getLayout(g, k_r, k_w, params['max_it'], params['patience'])
                
            data = [idx, k_r, k_w, params['patience'], params['max_it'], init_cross, best_cross, best_cross_it]
            
            res = res.append( {col_names[i]:data[i] for i in range(len(data))}, ignore_index=True )
            
            if (idx + 1) % params['save_each'] == 0:
                logger.info("Autosave")
                res.to_csv(f'{output_filepath}/autosave.csv')

    os.remove(f'{output_filepath}/autosave.csv')
    res.to_csv(f'{output_filepath}/exp{experiment_num}_res.csv')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()