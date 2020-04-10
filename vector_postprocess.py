import argparse
import numpy as np
from sklearn.decomposition import PCA


def main(args):
    """ vector postprocess

    :param wv: word vector 
    :param n_components: number of dimensions postprocessing
    
    :return: postprocessed vector
    """
    wv = np.load(args.model_path)
    pca = PCA(n_components=args.n_components)
    mean = np.average(wv, axis=0)
    pca.fit(wv - mean)
    components = np.matmul(np.matmul(wv, pca.components_.T), pca.components_)
    processed = wv - mean - components
    np.save(f'WV_abtt-{args.n_components}', processed)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model')
    parser.add_argument('--n_components', type=int, default=5, help='number of dimensions postprocessing')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
