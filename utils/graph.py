import numpy as np
from .dla import get_neighbors

def encode_positions(positions, T=150, harmonics=3):
    x, y = positions[:, 0], positions[:, 1]
    x_embed = np.concatenate(
        [np.sin(2*np.pi*x/T*(i+1)) for i in range(harmonics)] + \
        [np.cos(2*np.pi*x/T*(i+1)) for i in range(harmonics)]
        ).reshape(-1, harmonics*2)
    y_embed = np.concatenate(
        [np.sin(2*np.pi*y/T*(i+1)) for i in range(harmonics)] + \
        [np.cos(2*np.pi*y/T*(i+1)) for i in range(harmonics)]
        ).reshape(-1, harmonics*2)
    return np.concatenate([x_embed, y_embed], axis=1)

def grid_to_graph(grid, distribution=None):
    neighbors = get_neighbors(grid)
    node_positions = np.argwhere(neighbors > 0)

    X1, X2 = np.meshgrid(node_positions[:,0], node_positions[:,0])
    Y1, Y2 = np.meshgrid(node_positions[:,1], node_positions[:,1])
    D = np.abs(X1 - X2) + np.abs(Y1 - Y2)
    edges = np.argwhere(D == 1)

    mask = np.logical_and(
        neighbors[node_positions[:, 0], node_positions[:, 1]] > 0, 
        grid[node_positions[:, 0], node_positions[:, 1]] == 0
        ).reshape(-1, 1)

    input_nodes = np.concatenate([encode_positions(node_positions), grid[node_positions[:, 0], node_positions[:, 1]].reshape(-1, 1)], axis=1)
    if distribution is None:
        return node_positions, edges, mask, input_nodes, None

    dist_grid = grid + distribution
    target_nodes = dist_grid[node_positions[:, 0], node_positions[:, 1]].reshape(-1, 1)
    return node_positions, edges, mask, input_nodes, target_nodes

def graph_to_grid(node_positions, nodes):
    grid = np.zeros((150, 150))
    for i, pos in enumerate(node_positions):
        grid[pos[0], pos[1]] = nodes[i, -1]
    return grid
    
