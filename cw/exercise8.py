import numpy as np
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets as skdata


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:
Report your scores here.

'''


def plot_scatters(X, colors, labels, markers, title, axis_names, plot_3d=False):
    '''
    Creates scatter plot

    Args:
        X : list[numpy]
            list of numpy arrays (must have 3 dimensions for 3d plot)
        colors : list[str]
            list of colors to use
        labels : list[str]
            list of labels for legends
        markers : list[str]
            list of markers to use
        axis_names : list[str]
            names of each axis
        title : str
            title of plot
        plot_3d : bool
            if set, creates 3d plot, requires 3d data
    '''

    # Make sure data matches colors, labels, markers
    assert len(X) == len(colors)
    assert len(X) == len(labels)
    assert len(X) == len(markers)

    # Make sure we have right data type and number of axis names
    if plot_3d:
        assert X[0].shape[1] == 3
        assert len(axis_names) == 3
    else:
        assert X[0].shape[1] == 2
        assert len(axis_names) == 2

    fig = plt.figure()
    fig.suptitle(title)

    if plot_3d:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.set_zlabel(axis_names[2])

        for x, c, l, m in zip(X, colors, labels, markers):
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c, label=l, marker=m)
            ax.legend(loc='best')
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])

        for x, c, l, m in zip(X, colors, labels, markers):
            ax.scatter(x[:, 0], x[:, 1], c=c, label=l, marker=m)
            ax.legend(loc='best')


'''
Implementation of Principal Component Analysis (PCA) for dimensionality reduction
'''
class PrincipalComponentAnalysis(object):

    def __init__(self, k):
        # Number of eigenvectors to keep
        self.__k = k

        # Mean of the dataset
        self.__mean = None

        # Linear weights or transformation to project to lower subspace
        self.__weights = None

    def __center(self, X):
        '''
        Centers the data to zero-mean

        Args:
            X : numpy
                N x d feature vector

        Returns:
            numpy : N x d centered feature vector
        '''

        # TODO: Center the data

        return np.zeros_like(X)

    def __covariance_matrix(self, X):
        '''
        Computes the covariance matrix of a feature vector

        Args:
            X : numpy
                N x d feature vector

        Returns:
            numpy : d x d covariance matrix
        '''

        # TODO: Compute the covariance matrix

        return np.zeros([X.shape[1], X.shape[1]])

    def __fetch_weights(self, C):
        '''
        Obtains the top k eigenvectors (weights) from a covariance matrix C

        Args:
            C : numpy
                d x d covariance matrix

        Returns:
            numpy : d x k eigenvectors
        '''

        # TODO: Obtain the top k eigenvectors

        return np.zeros_like(C)

    def project_to_subspace(self, X):
        '''
        Project data X to lower dimension subspace using the top k eigenvectors

        Args:
            X : numpy
                N x d covariance matrix
            k : int
                number of eigenvectors to keep

        Returns:
            numpy : N x k feature vector
        '''

        # TODO: Computes transformation to lower dimension and project to subspace

        return X

    def reconstruct_from_subspace(self, Z):
        '''
        Reconstruct the original feature vector from the latent vector

        Args:
            Z : numpy
                N x k latent vector

        Returns:
            numpy : N x d feature vector
        '''

        # TODO: Reconstruct the original feature vector

        return Z


if __name__ == '__main__':

    # Load the iris dataset 150 samples of 4 dimensions
    iris_dataset = skdata.load_iris()
    X_iris = iris_dataset.data
    y_iris = iris_dataset.target

    # Initialize plotting colors, labels and markers for iris dataset
    colors_iris = ('blue', 'red', 'green')
    labels_iris = ('Setosa', 'Versicolour', 'Virginica')
    markers_iris = ('o', '^', '+')

    # TODO: Initialize Principal Component Analysis instance for k = 3


    # TODO: Visualize iris dataset in 3 dimension


    # TODO: Initialize Principal Component Analysis instance for k = 2


    # TODO: Visualize iris dataset in 2 dimensions


    # Possible number of eigenvectors to keep
    K = [4, 3, 2, 1]

    # MSE scores to keep track of loss from compression
    mse_scores = []

    for k in K:
        # TODO: Initialize PrincipalComponentAnalysis instance for k

        # TODO: Project the data to subspace

        # TODO: Reconstruct the original data

        # TODO: Measures mean squared error between original data and reconstructed data
        mse_score = 0.0

        # Save MSE score
        mse_scores.append(mse_score)

    # Creat plot for MSE for reconstruction
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('Iris Dataset Reconstruction Loss')

    ax.plot(K, mse_scores, marker='o', color='b', label='MSE')
    ax.legend(loc='best')
    ax.set_xlabel('k')
    ax.set_ylabel('MSE')

    # Show plots
    plt.show()
