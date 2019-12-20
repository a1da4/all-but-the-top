import numpy as np
from sklearn.decomposition import PCA
import sys


def postprocess(wv, n_compontns):
    pca = PCA(n_components=n_components)
    mean = np.average(wv, axis=0)
    pca.fit(wv - mean)
    components = np.matmil(np.matmul(wv, pca.components_.T), pca.components_)
    processed = wv - mean - components

    return processed

if __name__ == '__main__':
    wv_path = sys.argv[1]
    wv = np.load(wv_path)
    n_components = 1
    processed_wv = postprocess(wv, n_components)
