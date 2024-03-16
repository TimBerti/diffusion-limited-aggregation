import numpy as np

def get_neighbors(grid):
    return np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)

def new_position(grid):
    n = grid.shape[0]
    x, y = n//2, n//2
    while grid[x, y] == 1:
        theta = np.random.rand() * 2 * np.pi
        x = int(n//2 * np.cos(theta) + n//2)
        y = int(n//2 * np.sin(theta) + n//2)
    return np.array([x, y])

def step(grid, current, neighbors):
    n = grid.shape[0]
    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    direction = directions[np.random.randint(0, 4)]
    new = current + direction
    new = np.mod(new, n)
    if neighbors[*new] > 0:
        return new, True
    return new, False

def radius_of_gyration(grid):
    x, y = np.where(grid == 1)
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.mean(x**2 + y**2)