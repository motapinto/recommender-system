import warnings
import numpy as np
from enum import Enum
import scipy.sparse as sps
from Recommenders.Similarity.Compute_Similarity_Euclidean import Compute_Similarity_Euclidean
from Recommenders.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython

class SimilarityFunction(Enum):
    DICE = 'dice'
    TVERSKY = 'tversky'
    JACCARD = 'jaccard'
    TANIMOTO = 'tanimoto'
    ADJUSTED = 'adjusted'
    COSINE = 'cosine'
    PEARSON = 'pearson'
    ASYMMETRIC = 'asymmetric'
    EUCLIDEAN = 'euclidean'

class Compute_Similarity:
    def __init__(self, dataMatrix, similarity=None, **args):
        '''
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:              scipy sparse matrix |features|x|items| or |users|x|items|
        :param use_implementation:      'density' will choose the most efficient implementation automatically
                                        'cython' will use the cython implementation, if available. Most efficient for sparse matrix
                                        'python' will use the python implementation. Most efficient for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        '''

        assert np.all(np.isfinite(dataMatrix.data)), \
            'Compute_Similarity: Data matrix contains {} non finite values'.format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        self.dense = False

        if similarity == 'euclidean':
            self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)

        else:
            columns_with_full_features = np.sum(np.ediff1d(sps.csc_matrix(dataMatrix).indptr) == dataMatrix.shape[0])

            if similarity in ['dice', 'jaccard', 'tversky'] and columns_with_full_features >= dataMatrix.shape[1]/2:
                warnings.warn(
                    'Compute_Similarity: {:.2f}% of the columns have all features, '
                    'set-based similarity heuristics will not be able to discriminate between the columns.'.format(columns_with_full_features/dataMatrix.shape[1]*100))

            if dataMatrix.shape[0] == 1 and columns_with_full_features >= dataMatrix.shape[1]/2:
                warnings.warn(
                    'Compute_Similarity: {:.2f}% of the columns have a value for the single feature the data has, '
                    'most similarity heuristics will not be able to discriminate between the columns.'.format(columns_with_full_features/dataMatrix.shape[1]*100))

            assert not (dataMatrix.shape[0] == 1 and dataMatrix.nnz == dataMatrix.shape[1]),\
                'Compute_Similarity: data has only 1 feature (shape: {}) with values in all columns,' \
                ' cosine and set-based similarities are not able to discriminate 1-dimensional dense data,' \
                ' use Euclidean similarity instead.'.format(dataMatrix.shape)

            if similarity is not None:
                args['similarity'] = similarity

            try:
                self.compute_similarity_object = Compute_Similarity_Cython(dataMatrix, **args)
            except ImportError:
                print('Unable to load Cython Compute_Similarity')

    def compute_similarity(self,  **args):
        return self.compute_similarity_object.compute_similarity(**args)
