import argparse
import numpy as np
from sklearn.decomposition import PCA


def main(args):
    """ vector postprocess

    :param wv: word vector 
    :param n_components: number of dimensions postprocessing
    
    :return: postprocessed vector
    """
    if args.is_word2vec:
        # gensim.models Word2Vec model
        from gensim.models import Word2Vec
        model = Word2Vec.load(args.model_path)
        wv = model.wv.syn0
    else:
        # numpy model
        wv = np.load(args.model_path)

    print(args.is_word2vec)

    pca = PCA(n_components=args.n_components)
    mean = np.average(wv, axis=0)
    pca.fit(wv - mean)
    components = np.matmul(np.matmul(wv, pca.components_.T), pca.components_)
    processed = wv - mean - components
    if args.is_word2vec:
        model.wv.syn0 = processed
        model.save(f'{args.model_path}_abtt-{args.n_components}')
    else:
        np.save(f'{args.model_path[:-4]}_abtt-{args.n_components}.npy', processed)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', help='path of model')
    parser.add_argument('-n', '--n_components', type=int, default=5, help='number of dimensions postprocessing')
    parser.add_argument('-w', '--is_word2vec', action='store_true', help='if the model is word2vec in gensim, please call')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
