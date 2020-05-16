import argparse
import numpy as np
from sklearn.decomposition import PCA

def main(model_path, n_components):
    """ vector postprocess

    :param wv: word vector 
    :param n_components: number of dimensions postprocessing
    
    :return: postprocessed vector
    """
    try:
        # gensim.models Word2Vec model
        from gensim.models import Word2Vec
        model = Word2Vec.load(model_path)
        wv = model.wv.syn0
        is_word2vec = True
    except:
        # numpy model
        wv = np.load(model_path)
        is_word2vec = False

    print(f'is_word2vec: {is_word2vec}')

    pca = PCA(n_components=n_components)
    mean = np.average(wv, axis=0)
    pca.fit(wv - mean)
    components = np.matmul(np.matmul(wv, pca.components_.T), pca.components_)
    processed = wv - mean - components
    if is_word2vec:
        model.wv.syn0 = processed
        model.save(f'{model_path}_abtt-{n_components}')
    else:
        np.save(f'{model_path[:-4]}_abtt-{n_components}.npy', processed)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', nargs='*', help='path of model(s)')
    parser.add_argument('-n', '--n_components', type=int, default=5, help='number of dimensions postprocessing')

    args = parser.parse_args()
    for model_path in args.model_path:
        main(model_path, args.n_components)


if __name__ == '__main__':
    cli_main()
