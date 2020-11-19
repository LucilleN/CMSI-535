import numpy as np
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets as skdata


'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

Summary:

MSE score with k=4: 0.0000
MSE score with k=3: 0.0059
MSE score with k=2: 0.0253
MSE score with k=1: 0.0856

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

        # Center the data
        # B = X - mu

        mu = np.mean(X, axis=0)
        self.__mean = mu

        B = X - mu

        return B

    def __covariance_matrix(self, X):
        '''
        Computes the covariance matrix of a feature vector

        Args:
            X : numpy
                N x d feature vector

        Returns:
            numpy : d x d covariance matrix
        '''

        # Compute the covariance matrix
        
        N = X.shape[0]

        C = 1 / (N-1) * np.matmul(X.T, X)

        return C

    def __fetch_weights(self, C):
        '''
        Obtains the top k eigenvectors (weights) from a covariance matrix C

        Args:
            C : numpy
                d x d covariance matrix

        Returns:
            numpy : d x k eigenvectors (this is W)
        '''

        # Obtain the top k eigenvectors
        
        # Make sure that k is less than or equal to d 
        # (d here is D on the slides, k is little d on the slides)
        assert self.__k <= C.shape[0]

        # Eigen decomposition: V^{-1} C V = \Sigma V
        S, V = np.linalg.eig(C)

        # S is singular values of eigenvalues
        # We want to sort them in descending order
        # and we care about the positions of the new ordering
        # Use np.argsort, which sorts the indexes in ascending order 
        # We want descending order, so we reverse it with [::-1]
        # order contains the sorted indices of the eigenvalues, 
        # which corresponds to the eigenvectors
        order = np.argsort(S)[::-1]

        # select the top k eigenvectors
        # V[:, order] rearranges V from largest to smallest based on S, the eigenvalues
        # Grab from 0 up to k eigenvectors
        # now W is (d x k)
        # This is the latent vector we want to learn
        W = V[:, order][:, 0:self.__k]

        return W

    def project_to_subspace(self, X):
        '''
        Project data X to lower dimension subspace using the top k eigenvectors

        Args:
            X : numpy
                N x d covariance matrix
            k : int
                number of eigenvectors to keep

        Returns:
            numpy : N x k feature vector (this is Z)
        '''

        # Computes transformation to lower dimension and project to subspace

        # 1. Center the data
        B = self.__center(X)

        # 2. Compute the covariance matrix
        C = self.__covariance_matrix(B)

        # 3. Find the weights (W) that take us from d to k dimensions (fetch_weights)
        #    and set them to self.__weights
        W = self.__fetch_weights(C)
        self.__weights = W

        # 4. Project X down to k dimensions using the weights (W) to yield Z: Z = BW
        Z = np.matmul(B, W)

        # 5. Return Z
        return Z

    def reconstruct_from_subspace(self, Z):
        '''
        Reconstruct the original feature vector from the latent vector

        Args:
            Z : numpy
                N x k latent vector

        Returns:
            numpy : N x d feature vector (returns X hat)
        '''

        # Reconstruct the original feature vector
        # X^hat = Z W.T + mu
        X_hat = np.matmul(Z, self.__weights.T) + self.__mean

        return X_hat


if __name__ == '__main__':

    # Load the iris dataset 150 samples of 4 dimensions
    iris_dataset = skdata.load_iris()
    X_iris = iris_dataset.data
    y_iris = iris_dataset.target

    # Initialize plotting colors, labels and markers for iris dataset
    colors_iris = ('blue', 'red', 'green')
    labels_iris = ('Setosa', 'Versicolour', 'Virginica')
    markers_iris = ('o', '^', '+')

    # Visualize the iris dataset by truncating the last dimension
    
    # Iris dataset is (150 x 4) = (N x d), so we remove the last dimension
    X_iris_trunc = X_iris[:, 0:3]

    # Find every sample of X that belongs to class 0, class 1, and class 2 separately
    # Together N_class0 + N_class1 + N_class2 = N
    X_iris_trunc_class_split = [
        # This grabs (N_class0 x 3)
        X_iris_trunc[np.where(y_iris == 0)[0], :],
        # This grabs (N_class1 x 3)
        X_iris_trunc[np.where(y_iris == 1)[0], :],
        # This grabs (N_class2 x 3)
        X_iris_trunc[np.where(y_iris == 2)[0], :]
    ]

    plot_scatters(
        X=X_iris_trunc_class_split,
        colors=colors_iris,
        labels=labels_iris,
        markers=markers_iris,
        title='Iris Dataset Truncated By Last Dimension',
        axis_names=['x1', 'x2', 'x3'],
        plot_3d=True
    )
    
    # Initialize Principal Component Analysis instance for k = 3
    pca_k3 = PrincipalComponentAnalysis(k=3)
    Z_k3 = pca_k3.project_to_subspace(X_iris)

    # Visualize iris dataset in 3 dimension
    X_iris_pca_k3_class_split = [
        # This grabs (N_class0 x 3)
        Z_k3[np.where(y_iris == 0)[0], :],
        # This grabs (N_class1 x 3)
        Z_k3[np.where(y_iris == 1)[0], :],
        # This grabs (N_class2 x 3)
        Z_k3[np.where(y_iris == 2)[0], :]
    ]
    plot_scatters(
        X=X_iris_pca_k3_class_split,
        colors=colors_iris,
        labels=labels_iris,
        markers=markers_iris,
        title='Iris Dataset with PCA for k=3',
        axis_names=['x1', 'x2', 'x3'],
        plot_3d=True
    )

    # Initialize Principal Component Analysis instance for k = 2
    pca_k2 = PrincipalComponentAnalysis(k=2)
    Z_k2 = pca_k2.project_to_subspace(X_iris)

    # Visualize iris dataset in 2 dimensions
    X_iris_pca_k2_class_split = [
        # This grabs (N_class0 x 2)
        Z_k2[np.where(y_iris == 0)[0], :],
        # This grabs (N_class1 x 2)
        Z_k2[np.where(y_iris == 1)[0], :],
        # This grabs (N_class2 x 2)
        Z_k2[np.where(y_iris == 2)[0], :]
    ]
    plot_scatters(
        X=X_iris_pca_k2_class_split,
        colors=colors_iris,
        labels=labels_iris,
        markers=markers_iris,
        title='Iris Dataset with PCA for k=2',
        axis_names=['x1', 'x2'],
        plot_3d=False
    )

    # Possible number of eigenvectors to keep
    K = [4, 3, 2, 1]

    # MSE scores to keep track of loss from compression
    mse_scores = []

    for k in K:
        # Initialize PrincipalComponentAnalysis instance for k
        pca = PrincipalComponentAnalysis(k=k)

        # Project the data to subspace
        Z = pca.project_to_subspace(X_iris)

        # Reconstruct the original data
        X_hat = pca.reconstruct_from_subspace(Z)

        # Measures mean squared error between original data and reconstructed data
        mse_score = skmetrics.mean_squared_error(X_hat, X_iris)
        print("MSE score with k={}: {:.4f}".format(k, mse_score))

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
