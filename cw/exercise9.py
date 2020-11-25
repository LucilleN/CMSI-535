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

In this exercise, we visualized the MSE loss on reconstructed datasets created with
PCAs using values of k from 1 to 64 on the original digits dataset. We then used PCA 
to generate new samples of synthesized handwritten 2's. 

'''


def plot_images(X, n_row, n_col, title):
    '''
    Creates n_row by n_col panel of images

    Args:
        X : numpy
            N x h x w numpy array
        n_row : int
            number of rows in figure
        n_col : list[str]
            number of columns in figure
        title : str
            title of plot
    '''

    fig = plt.figure()
    fig.suptitle(title)

    for i in range(1, n_row * n_col + 1):

        ax = fig.add_subplot(n_row, n_col, i)

        x_i = X[i, ...]
        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, axis=0)

        ax.imshow(x_i)

        plt.box(False)
        plt.axis('off')


def compute_eigenvalues(X, sorted=False):
    '''
    Computes eigenvalues for a dataset X

    Args:
        X : numpy
            N x d numpy array
        sorted : bool
            if set, then sort eigenvalues descending

    Returns:
        S : numpy
            d element array
    '''

    # Compute eigenvalues of a dataset X

    # Compute the mean along axis 0 (axis 0 refers to the # of dif data samples)
    X_mean = np.mean(X, axis=0)

    # Center the data
    B = X - X_mean

    # Compute covariance: C = 1/(N-1) B^T B
    N = X.shape[0]
    C = 1.0 / (N - 1) * np.matmul(B.T, B)

    # Eigen decomposition
    S, _ = np.linalg.eig(C)

    if sorted:
        # Sort the eigenvalues in descending order
        order = np.argsort(S)[::-1]

        S = S[order]

    return S


def plot_eigenvalues(X):
    '''
    Plots eigenvalues for a dataset X

    Args:
        X : numpy
            N x d numpy array
    '''

    # Compute eigenvalues and sort them
    S_sorted = compute_eigenvalues(X, sorted=True)

    # Plot eigenvales in descending order
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(S_sorted)

    fig.suptitle('Sorted Eigenvalues')

    ax.set_xlabel('Eigenvectors')
    ax.set_ylabel('Eigenvalues')


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

        # Eigenvalues of the dataset
        self.__eigenvalues = None

    def __center(self, X):
        '''
        Centers the data to zero-mean

        Args:
            X : numpy
                N x d feature vector

        Returns:
            numpy : N x d centered feature vector
        '''

        # Compute the mean
        X_mean = np.mean(X, axis=0)

        # Store the mean to use during reconstruction
        self.__mean = X_mean

        # Center the data
        B = X - X_mean
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
        C = 1.0 / (N - 1) * np.matmul(X.T, X)

        return C

    def __fetch_weights(self, C):
        '''
        Obtains the top k eigenvectors (weights) from a covariance matrix C

        Args:
            C : numpy
                d x d covariance matrix

        Returns:
            numpy : d x k eigenvectors
        '''

        # Obtain the top k eigenvectors

        # Make sure that we have k less than d (the shape of C is d x d)
        assert self.__k <= C.shape[0]

        # Eigen decomposition
        S, V = np.linalg.eig(C)

        # Sort the eigenvalues in descending order
        order = np.argsort(S)[::-1]

        # Store the sorted eigenvalues for generating new samples
        self.__eigenvalues = S[order]

        W = V[:, order][:, 0:self.__k]

        # Store weights
        self.__weights = W

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
            numpy : N x k feature vector
        '''

        # Computes transformation to lower dimension and project to subspace

        # Center the data
        B = self.__center(X)

        # Compute covariance matrix
        C = self.__covariance_matrix(B)

        # Obtain the weights to project to subspace
        W = self.__fetch_weights(C)

        # Project B to subspace using the weights
        Z = np.matmul(B, W)

        return Z

    def reconstruct_from_subspace(self, Z):
        '''
        Reconstruct the original feature vector from the latent vector

        Args:
            Z : numpy
                N x k latent vector

        Returns:
            numpy : N x d feature vector
        '''

        # Reconstruct the original feature vector
        X_hat = np.matmul(Z, self.__weights.T) + self.__mean

        return X_hat

    def generate_new_samples(self, Z, n_sample):
        '''
        Generates new data points by sampling from Z

        Args:
            Z : numpy
                N x k latent vector
            n_sample: int

        Returns:
            numpy : N x d feature vector
        '''

        # Generate new samples

        # We need the eigenvalues so we can sample from N(0, sigma^2)
        # Eigenvalues S = sigma^2

        # np.random.normal(mean, stdev, shape)
        # stdev = sqrt(sigma^2)
        # We want 9 samples of k dimensions: (n_sample x k)
        Z_sample_digits_2s = np.random.normal(
            0.0, np.sqrt(self.__eigenvalues), (n_sample, Z.shape[1]))

        # This gives us X_hat which is (n_sample x d) since Z is (n_sample x k) and we project
        # back from k dimensions to d dimensions
        X_hat = self.reconstruct_from_subspace(Z_sample_digits_2s)

        return X_hat


if __name__ == '__main__':

    # Load the digits dataset 1797 samples of 8 by 8 dimensions
    digits_dataset = skdata.load_digits()
    X_digits = digits_dataset.data
    y_digits = digits_dataset.target

    '''
    PCA for handwritten digits reconstruction
    '''

    # Compute the number of dimensions in the dataset: (N, 8, 8) or (N, 8 * 8) = (N, 64)
    # Multiply the dimensions of shape everywhere but in 0th index
    # 8 * 8 = 64
    n_dim = np.prod(X_digits.shape[1:])

    # Reshape handwritten digits dataset to (N, 8, 8)
    X_digits = np.reshape(X_digits, (-1, 8, 8))

    # Plot 3 x 3 panel of handwritten digits and name is 'Handwritten digits dataset'
    plot_images(
        X=X_digits,
        n_row=3,
        n_col=3,
        title='Handwritten digits dataset'
    )

    # Vectorize handwritten digits dataset: (N x 8 x 8) -> (N x 64)
    X_digits = np.reshape(X_digits, (-1, n_dim))

    # Plot 9 x 1 panel of handwritten digits and name is 'Vectorized handwritten digits dataset'
    plot_images(
        X=X_digits,
        n_row=9,
        n_col=1,
        title='Handwritten digits dataset'
    )

    # Plot eigenvalues for digits dataset
    plot_eigenvalues(X_digits)

    # Select K from 1 to max number of dimensions
    K = range(1, n_dim + 1)  # [1, 2, 3, ... , 64]

    # MSE scores to keep track of loss from compression
    mse_scores = []

    for k in K:
        # Initialize PrincipalComponent
        pca = PrincipalComponentAnalysis(k=k)

        # Project the data to subspace
        Z_digits = pca.project_to_subspace(X_digits)

        # Reconstruct the original data
        X_hat_digits = pca.reconstruct_from_subspace(Z_digits)

        # Measures mean squared error between original data and reconstructed data
        mse_score = skmetrics.mean_squared_error(X_digits, X_hat_digits)

        # Save MSE score
        mse_scores.append(mse_score)

    # Creat plot for MSE for reconstruction
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('Digits Dataset Reconstruction Loss')

    ax.plot(K, mse_scores, marker='o', color='b', label='MSE')
    ax.legend(loc='best')
    ax.set_xlabel('k')
    ax.set_ylabel('MSE')

    '''
    PCA as a generative model
    '''

    # Select all 2's from the digit dataset
    idx = np.where(y_digits == 2)[0]
    X_digits_2s = X_digits[idx, :]

    # Reshape handwritten digits dataset of 2s to (N, 8, 8)
    X_digits_2s = np.reshape(X_digits_2s, (-1, 8, 8))

    # Plot 3 x 3 panel of handwritten digits and name is 'Handwritten digits dataset of 2s'
    plot_images(
        X=X_digits_2s,
        n_row=3,
        n_col=3,
        title='Handwritten digits dataset of 2s'
    )

    # Vectorize handwritten digits dataset: (N x 64)
    X_digits_2s = np.reshape(X_digits_2s, (-1, n_dim))

    # Compute PCA and keep all principal components
    pca_digits_2s = PrincipalComponentAnalysis(k=n_dim)

    # Sample from subspace to generate a new Z vector of 9 data points
    Z_digits_2s = pca_digits_2s.project_to_subspace(X_digits_2s)

    # Generate novel samples
    X_hat_digits_2s = pca_digits_2s.generate_new_samples(
        Z_digits_2s, n_sample=20)
    X_hat_digits_2s = np.reshape(X_hat_digits_2s, (-1, 8, 8))

    # Plot 3 x 3 panel of handwritten digits of newly generated data
    plot_images(
        X=X_hat_digits_2s,
        n_row=3,
        n_col=3,
        title='Synthesized handwritten digits dataset of 2s'
    )

    # Show plots
    plt.show()
