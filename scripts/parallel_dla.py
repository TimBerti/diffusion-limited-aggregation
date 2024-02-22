import uuid
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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

    grid_list = [grid.reshape(-1)]
    i = 0
    while i < aggragations:
        grid, current, neighbors, stopped = step(grid, current, neighbors)
        if stopped:
            i += 1
            grid_list.append(grid.copy().reshape(-1))

    df = pd.DataFrame(np.array(grid_list))
    df.to_parquet(f'./data/size{size}_aggregations{aggragations}_{str(uuid.uuid4())}.parquet')


if __name__ == '__main__':
    Parallel(n_jobs=-1, verbose=10)(delayed(run)() for _ in range(runs))