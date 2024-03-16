import uuid
import h5py
import os

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

data_dir = './data'
runs = 1000
aggragations = 1000
size = 150
directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

def get_neighbors(grid):
    return np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)

def new_position(grid):
    x, y = size//2, size//2
    while grid[x, y] == 1:
        theta = np.random.rand() * 2 * np.pi
        x = int(size//2 * np.cos(theta) + size//2)
        y = int(size//2 * np.sin(theta) + size//2)
    return np.array([x, y])

def step(grid, current, neighbors):
    direction = directions[np.random.randint(0, 4)]
    new = current + direction
    new = np.mod(new, size)
    if neighbors[*new] > 0:
        grid[*new] = 1
        neighbors = get_neighbors(grid)
        return grid, new_position(grid), neighbors, True
    return grid, new, neighbors, False

def run():
    grid = np.zeros((size,size))
    grid[size//2, size//2] = 1

    current = new_position(grid)
    neighbors = get_neighbors(grid)

    grid_list = [grid.copy().reshape(-1)]
    i = 0
    while i < aggragations:
        grid, current, neighbors, stopped = step(grid, current, neighbors)
        if stopped:
            i += 1
            grid_list.append(grid.copy().reshape(-1))

    df = pd.DataFrame(np.array(grid_list))
    df.to_parquet(os.path.join(data_dir, f'size{size}_aggregations{aggragations}_{str(uuid.uuid4())}.parquet'))

def convert_to_hdf5():
    file_names = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    size = len(file_names) * 1000
    with h5py.File(os.path.join(data_dir, 'images_diffs.hdf5'), 'w') as h5f:
        images_dset = h5f.create_dataset('images', (size, 1, 150, 150), dtype='float32')
        diffs_dset = h5f.create_dataset('diffs', (size,), dtype='int')
        
        idx = 0
        for file_name in tqdm(file_names):
            X = pd.read_parquet(os.path.join(data_dir, file_name)).to_numpy()
            diffs = X[1:] - X[:-1]
            diff_index = np.argmax(diffs, axis=1)
            diffs_dset[idx:idx + len(diffs)] = diff_index

            images = X.reshape(-1, 1, 150, 150)
            images_dset[idx:idx + len(diffs)] = images[:-1]
            idx = idx + len(diffs)

if __name__ == '__main__':
    Parallel(n_jobs=-1, verbose=10)(delayed(run)() for _ in range(runs))
    convert_to_hdf5()