import uuid
import os

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import os
import sys
sys.path.append(os.path.join(os.path.abspath(''), '.'))
from utils.dla import get_neighbors, new_start, next_aggregation, probability_distribution

data_dir = './data'
runs = 100
aggragations = 1000
size = 150

def run():
    grid = np.zeros((size,size))
    grid[size//2, size//2] = 1

    start = new_start(grid)
    neighbors = get_neighbors(grid)

    grid_list = []
    distribution_list = []

    for _ in range(aggragations):
        grid_list.append(grid.copy().reshape(-1))
        n_neighbors = ((neighbors + grid) > 0).sum() - (grid > 0).sum()
        distribution_list.append(probability_distribution(grid, np.sqrt(n_neighbors).astype(int) + 6).reshape(-1))

        aggregation_point = next_aggregation(start, neighbors)
        grid[*aggregation_point] = 1
        start = new_start(grid)
        neighbors = get_neighbors(grid)

    grid_df = pd.DataFrame(np.array(grid_list))
    distribution_df = pd.DataFrame(np.array(distribution_list))

    run_id = str(uuid.uuid4())
    grid_df.to_parquet(os.path.join(data_dir, f'{run_id}_grid.parquet'))
    distribution_df.to_parquet(os.path.join(data_dir, f'{run_id}_distribution.parquet'))

if __name__ == '__main__':
    Parallel(n_jobs=-1, verbose=10)(delayed(run)() for _ in range(runs))