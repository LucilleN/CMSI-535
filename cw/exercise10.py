'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:

TODO: Explain your design for your neural network e.g.
How many layers, neurons did you use? What kind of activation function did you use?
Please give reasoning of why you chose a certain number of neurons for some layers.

TODO: Report all of your hyper-parameters.


TODO: Report your scores here. Mean accuracy should exceed 54%

'''
import argparse
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

# Commandline arguments
parser.add_argument('--train_network',
    action='store_true', help='If set, then trains network')
parser.add_argument('--batch_size',
    type=int, default=4, help='Number of samples per batch')
parser.add_argument('--n_epoch',
    type=int, default=1, help='Number of times to iterate through dataset')
parser.add_argument('--learning_rate',
    type=float, default=1e-8, help='Base learning rate (alpha)')
parser.add_argument('--learning_rate_decay',
    type=float, default=0.50, help='Decay rate for learning rate')
parser.add_argument('--learning_rate_decay_period',
    type=float, default=1, help='Period before decaying learning rate')
parser.add_argument('--momentum',
    type=float, default=0.50, help='Momentum discount rate (beta)')
parser.add_argument('--lambda_weight_decay',
    type=float, default=0.0, help='Lambda used for weight decay')


args = parser.parse_args()


class NeuralNetwork(torch.nn.Module):
    '''
    Neural network class of fully connected layers
    '''

    def __init__(self, n_input_feature, n_output):
        super(NeuralNetwork, self).__init__()

        # TODO: Design your neural network

    def forward(self, x):
        '''
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        '''

        # TODO: Implement forward function

        return x

def train(net, dataloader, n_epoch, scheduler):
    '''
    Trains the network using a learning rate scheduler

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        scheduler : torch.optim.lr_scheduler
            https://pytorch.org/docs/stable/optim.html
            scheduler to adjust learning rate

    Returns:
        torch.nn.Module : trained network
    '''

    # TODO: Define cross entropy loss

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        for batch, (images, labels) in enumerate(dataloader):

            # TODO: Vectorize images from (N, H, W, C) to (N, d)

            # TODO: Forward through the network

            # TODO: Compute loss function

            # TODO: Update parameters by backpropagation

            # TODO: Update learning schedule

            # TODO: Clear gradients so we don't accumlate them from previous batches

            # TODO: Accumulate total loss for the epoch

            pass

        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch=%d  Loss: %.3f' % (epoch + 1, mean_loss))

    return net

def evaluate(net, dataloader, classes):
    '''
    Evaluates the network on a dataset

    Args:
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        classes : list[str]
            list of class names to be used in plot
    '''

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            # TODO: Vectorize images from (N, H, W, C) to (N, d)

            # TODO: Forward through the network

            # TODO: Take the argmax over the outputs

            # Accumulate number of samples
            n_sample = n_sample + labels.shape[0]

            # TODO: Check if our prediction is correct

    # TODO: Compute mean accuracy
    mean_accuracy = 0.0

    print('Mean accuracy over %d images: %d %%' % (n_sample, mean_accuracy))

    # TODO: Plot images with class names


def plot_images(X, n_row, n_col, fig_title, subplot_titles):
    '''
    Creates n_row by n_col panel of images

    Args:
        X : numpy
            N x h x w numpy array
        n_row : int
            number of rows in figure
        n_col : list[str]
            number of columns in figure
        fig_title : str
            title of plot
        subplot_titles : str
            title of subplot
    '''

    fig = plt.figure()
    fig.suptitle(fig_title)

    for i in range(1, n_row * n_col + 1):

        ax = fig.add_subplot(n_row, n_col, i)

        x_i = X[i, ...]

        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, axis=0)

        ax.set_title(subplot_titles[i])
        ax.imshow(x_i)

        plt.box(False)
        plt.axis('off')

if __name__ == '__main__':

    # Set up data preprocessing step
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    data_preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(16, 16)),
        torchvision.transforms.ToTensor()
    ])

    # Download and setup CIFAR10 training set
    cifar10_train = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=data_preprocess_transform)

    # Setup a dataloader (iterator) to fetch from the training set
    dataloader_train = torch.utils.data.DataLoader(
        cifar10_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

    # Download and setup CIFAR10 testing set
    cifar10_test = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=data_preprocess_transform)

    # Setup a dataloader (iterator) to fetch from the testing set
    dataloader_test = torch.utils.data.DataLoader(
        cifar10_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    # Define the possible classes in CIFAR10
    classes = [
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]

    n_input_feature = 16 * 16 * 3
    n_class = 10

    # TODO: Define network

    # TODO: Setup learning rate SGD optimizer and step function scheduler
    # https://pytorch.org/docs/stable/optim.html

    if args.train_network:
        # TODO: Set network to training mode

        # TODO: Train network and save into checkpoint

        pass
    else:
        # TODO: Load network from checkpoint

        pass

    # TODO: Set network to evaluation mode

    # TODO: Evaluate network on testing set
