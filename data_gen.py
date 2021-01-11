import os
import itertools
import functools
import collections

import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm

N_SAMPLES = [1000]
N_FEATURES = [1000]

Dataset = collections.namedtuple("Dataset", ["X", "y", "name"])

data_grid_1 = {
    "n_samples": N_SAMPLES,
    "n_features": [5000],
    "n_classes": [2],
}

data_grid_2 = {
    "n_samples": N_SAMPLES,
    "n_features": N_FEATURES,
    "n_classes": [2],
    "balance_ratio": [
        (0, 0),  # balanced
        (0.33, 0.67),
        (0.25, 0.75),
        (0.2, 0.8),
        (0.1, 0.9),
    ],
}

data_grid_3 = {
    "n_samples": N_SAMPLES,
    "n_features": N_FEATURES,
    "n_classes": [3],
    "balance_ratio": [
        (0, 0, 0),  # balanced
        (0.25, 0.25, 0.5),
        (0.2, 0.2, 0.6),
        (0.1, 0.1, 0.8),
        (0.05, 0.05, 0.9),
    ],
}


def main():
    dirname = "datasets"
    os.makedirs(dirname, exist_ok=True)
    print("saving datasets")
    for X, y, name in list(get_datasets()):
        filename = os.path.join(dirname, name + ".csv")
        data = np.concatenate((X, y[:, np.newaxis]), axis=1)
        print(f"creating file: {filename}")
        np.savetxt(
            filename,
            data,
            delimiter=",",
            fmt=["%.5f" for i in range(X.shape[1])] + ["%i"],
        )


def get_datasets():
    data = [product_from_dict(grid) for grid in [data_grid_1, data_grid_2, data_grid_3]]
    for data_info in tqdm(flatten(data)):
        n_redundant, n_informative = redundant_informative_split(
            data_info["n_features"]
        )
        balance_ratio = data_info.get("balance_ratio")
        weights = None
        if balance_ratio is None or balance_ratio[0] != 0:  # imbalanced
            weights = balance_ratio
        X, y = make_classification(
            n_classes=data_info["n_classes"],
            n_features=data_info["n_features"],
            n_informative=n_informative,
            n_redundant=n_redundant,
            random_state=1410,
            weights=weights,
        )
        name = make_dataset_name(data_info)
        yield Dataset(X, y, name)


def redundant_informative_split(n, redundant_ratio=0.2):
    n_redundant = int(n * redundant_ratio)
    n_informative = n - n_redundant
    return n_redundant, n_informative


def make_dataset_name(data_info):
    name = ""
    name += f"{data_info['n_features']}_features_"
    name += f"{data_info['n_classes']}_classes_"
    balance_ratio = data_info.get("balance_ratio")
    if balance_ratio is None or balance_ratio[0] == 0:
        name += "balanced"
    else:
        name += "_".join(str(x) for x in balance_ratio)
    return name


def map_key_to_every_value(key, values):
    return [{key: value} for value in values]


def merge_dicts(dicts):
    return functools.reduce(lambda a, b: {**a, **b}, dicts)


def product_from_dict(grid):
    """
    return list of dict with combinations of items from dict iterators values
    e.g.
        grid = {
            'even': [2,4],
            'odd': [1,3]
        }
        returns:
        [
            {'even': 2, 'odd': 1},
            {'even': 2, 'odd': 3},
            {'even': 4, 'odd': 1},
            {'even': 4, 'odd': 3}
        ]
    """
    buff = [map_key_to_every_value(key, value) for key, value in grid.items()]
    return [merge_dicts(args) for args in itertools.product(*buff)]


def flatten(iterable):
    return list(itertools.chain.from_iterable(iterable))


if __name__ == "__main__":
    main()
