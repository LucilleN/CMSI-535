'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

Summary:

    In this exercise, we used PyTorch to design a Neural Network class by choosing how many
    layers it would have, how many neurons per layer, and what activation function(s) to use. 
    We then implemented functions to train the network using a variety of hyperparameters, as 
    well as to evaluate the network and display its predictions of some images. In our main 
    method, we implemented a training loop that trains the network over n_epochs, then
    evaluates it on the testing set. We tuned the network by running this file with different 
    combinations of the hyperparameter values (reported below).

Explain your design for your neural network e.g.
How many layers, neurons did you use? What kind of activation function did you use?
Please give reasoning of why you chose a certain number of neurons for some layers.

    My neural network has 5 hidden layers and 1 output layer, all of which are linear. 
    The first layer takes in n_input_features and has 1024 neurons, the second has 512, 
    the third has 256, the fourth has 128, and the fifth has 64 neurons; the final output 
    layer takes in those 64 inputs and outputs n_output dimensions. I chose to make the 
    first layer have 1024 dimensions because the input is made of 16x16 images with 3 
    color channels per pixel, which means that each sample has 16 x 16 x 3 = 768 features.
    I wanted to do some feature expansion on this input by having a layer with more 
    output neurons than inputs, rather than using polynomial feature expansion, because 
    going from 768 to 1024 features increases the dimensionality. Then for the rest of 
    the layers, I gradually decreased the number of neurons so that there wouldn't be any
    dramatic reductions in dimensionality that might result in too much information loss. 
    For all layers, I used the Leaky ReLu activation function.

Report all of your hyper-parameters.

    Run this command to train the network:
        python3 exercise10.py --train_network --batch_size 8 --n_epoch 60 --learning_rate 0.01 --lambda_weight_decay 0.0001 --learning_rate_decay 0.9 --learning_rate_decay_period 2 

    The hyperparameters are:
        - batch size: 8 samples
        - number of epochs: 60
        - learning rate: 0.01
        - learning rate decay: 0.9 (learning rate becomes 90% of what it was before, every 2 epochs)
        - learning rate decay period: every 2 epochs
        - momentum: 0.9 
        - lambda weight decay: 0.0001

Report your scores here. Mean accuracy should exceed 54%
    
    Epoch=1  Loss: 2.102
    Epoch=2  Loss: 1.799
    Epoch=3  Loss: 1.669
    ...
    Epoch=58  Loss: 0.012
    Epoch=59  Loss: 0.011
    Epoch=60  Loss: 0.010
    Mean accuracy over 10000 images: 55 %

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

        # Design your neural network

        # This gets 55% !!!
        # python3 exercise10.py --train_network --batch_size 8 --n_epoch 100 --learning_rate 0.01 --lambda_weight_decay 0.0 --learning_rate_decay 0.9 --learning_rate_decay_period 2 
        self.fully_connected_layer_1 = torch.nn.Linear(n_input_feature, 1024)
        self.fully_connected_layer_2 = torch.nn.Linear(1024, 512)
        # self.fully_connected_layer_1 = torch.nn.Linear(n_input_feature, 512)
        # self.fully_connected_layer_2 = torch.nn.Linear(512, 512)
        self.fully_connected_layer_3 = torch.nn.Linear(512, 256)
        self.fully_connected_layer_4 = torch.nn.Linear(256, 128)
        self.fully_connected_layer_5 = torch.nn.Linear(128, 64)
        
        self.output = torch.nn.Linear(64, n_output)

        self.activation_function = torch.nn.functional.leaky_relu


    def forward(self, x):
        '''
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        '''

        # Implement forward function
        x1 = self.fully_connected_layer_1(x)
        theta_x1 = self.activation_function(x1)

        x2 = self.fully_connected_layer_2(theta_x1)
        theta_x2 = self.activation_function(x2)

        x3 = self.fully_connected_layer_3(theta_x2)
        theta_x3 = self.activation_function(x3)

        x4 = self.fully_connected_layer_4(theta_x3)
        theta_x4 = self.activation_function(x4)

        x5 = self.fully_connected_layer_5(theta_x4)
        theta_x5 = self.activation_function(x5)

        output = self.output(theta_x5)

        return output


def train(net,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period):
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
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights
        learning_rate_decay : float
            rate of learning rate decay
        learning_rate_decay_period : int
            period to reduce learning rate based on decay e.g. every 2 epoch

    Returns:
        torch.nn.Module : trained network
    '''

    # Define cross entropy loss
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_decay * param_group['lr']

        for batch, (images, labels) in enumerate(dataloader):

            # Vectorize images from (N, H, W, C) to (N, d)
            n_dim = np.prod(images.shape[1:])
            images = images.view(-1, n_dim)

            # Forward through the network
            outputs = net(images)

            # Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # Compute loss function
            loss = loss_function(outputs, labels)

            # Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate total loss for the epoch
            total_loss = total_loss + loss.item()

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
    og_shape = None
    predictions = None

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            og_shape = images.shape

            # Vectorize images from (N, H, W, C) to (N, d)
            n_dim = np.prod(og_shape[1:])
            images = images.view(-1, n_dim)

            # Forward through the network
            outputs = net(images)

            # Take the argmax over the outputs
            _, predictions = torch.max(outputs, dim=1)

            # Accumulate number of samples
            n_sample = n_sample + labels.shape[0]

            # Check if our prediction is correct; if so, increment the number 
            # correct so we can check the accuracy later
            n_correct = n_correct + torch.sum(predictions == labels).item()

    # Compute mean accuracy as a percentage
    mean_accuracy = n_correct / n_sample * 100.0
    print('Mean accuracy over %d images: %d %%' % (n_sample, mean_accuracy))

    # Convert the last batch of images back to original shape
    images = images.view(og_shape[0], og_shape[1], og_shape[2], og_shape[3])
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # Map the last batch of predictions to their corresponding class labels
    prediction_classes = [classes[integer_label] for integer_label in predictions]

    # Plot images with class names
    plot_images(
        X=images,
        n_row=2,
        n_col=2,
        fig_title='Image Classification Predictions with PyTorch Neural Network',
        subplot_titles=prediction_classes
    )


def plot_images(X, n_row, n_col, fig_title, subplot_titles):
    '''
    Creates n_row by n_col panel of images

    Args:
        X : numpy
            N x h x w numpy array
        n_row : int
            number of rows in figure
        n_col : int
            number of columns in figure
        fig_title : str
            title of plot
        subplot_titles : list[str]
            title of subplot
    '''

    fig = plt.figure()
    fig.suptitle(fig_title)

    for i in range(1, n_row * n_col + 1):
    # for i in range(0, n_row * n_col):

        ax = fig.add_subplot(n_row, n_col, i)

        index = i-1
        x_i = X[index, ...]
        subplot_title_i = subplot_titles[index]

        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, axis=0)

        ax.set_title(subplot_title_i)
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

    # Define network
    net = NeuralNetwork(
        n_input_feature=n_input_feature,
        n_output=n_class)

    # Setup learning rate SGD optimizer and step function scheduler
    # https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.lambda_weight_decay, 
        momentum=args.momentum
    )

    if args.train_network:
        # Set network to training mode
        net.train()

        # Train network and save into checkpoint
        net = train(
            net=net,
            dataloader=dataloader_train,
            n_epoch=args.n_epoch,
            optimizer=optimizer,
            learning_rate_decay=args.learning_rate_decay,
            learning_rate_decay_period=args.learning_rate_decay_period)
        torch.save({ 'state_dict' : net.state_dict()}, './checkpoint.pth')
    else:
        # Load network from checkpoint
        checkpoint = torch.load('./checkpoint.pth')
        net.load_state_dict(checkpoint['state_dict'])

    # Set network to evaluation mode
    net.eval()

    # Evaluate network on testing set
    evaluate(
        net=net,
        dataloader=dataloader_test,
        classes=classes)

    plt.show()