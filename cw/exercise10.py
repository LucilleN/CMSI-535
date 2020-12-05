'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

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

        # Design your neural network
        self.fully_connected_layer_1 = torch.nn.Linear(n_input_feature, 1024)
        self.fully_connected_layer_2 = torch.nn.Linear(1024, 512)
        self.fully_connected_layer_3 = torch.nn.Linear(512, 256)
        self.fully_connected_layer_4 = torch.nn.Linear(256, 128)
        self.fully_connected_layer_5 = torch.nn.Linear(128, 64)
        self.fully_connected_layer_6 = torch.nn.Linear(64, 32)
        self.fully_connected_layer_7 = torch.nn.Linear(32, 16)
        self.fully_connected_layer_8 = torch.nn.Linear(16, 8)
        self.fully_connected_layer_9 = torch.nn.Linear(8, 8)
        self.fully_connected_layer_10 = torch.nn.Linear(8, 8)
        
        self.output = torch.nn.Linear(8, n_output)

        self.activation_function = torch.nn.functional.leaky_relu

        # self.fully_connected_layer_1 = torch.nn.Linear(n_input_feature, 16)
        # self.fully_connected_layer_2 = torch.nn.Linear(16, 12)
        # self.fully_connected_layer_3 = torch.nn.Linear(12, 8)
        # self.fully_connected_layer_4 = torch.nn.Linear(8, 8)
        # self.output = torch.nn.Linear(8, n_output)
        # self.activation_function = torch.nn.functional.relu
        

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

        x6 = self.fully_connected_layer_6(theta_x5)
        theta_x6 = self.activation_function(x6)

        x7 = self.fully_connected_layer_7(theta_x6)
        theta_x7 = self.activation_function(x7)

        x8 = self.fully_connected_layer_8(theta_x7)
        theta_x8 = self.activation_function(x8)

        x9 = self.fully_connected_layer_9(theta_x8)
        theta_x9 = self.activation_function(x9)

        x10 = self.fully_connected_layer_10(theta_x9)
        theta_x10 = self.activation_function(x10)

        output = self.output(theta_x10)
        # output = self.output(theta_x4)

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

    # TODO: Convert the last batch of images back to original shape
    images = images.view(og_shape[0], og_shape[1], og_shape[2], og_shape[3])
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # TODO: Map the last batch of predictions to their corresponding class labels
    # images_class_split = [images[np.where(predictions == label)[0], :] for label in range(len(classes))]
    #     # This grabs (N_class0 x 3)
    #     predictions[np.where(y_iris == 0)[0], :],
    #     # This grabs (N_class1 x 3)
    #     predictions[np.where(y_iris == 1)[0], :],
    #     # This grabs (N_class2 x 3)
    #     predictions[np.where(y_iris == 2)[0], :]
    # ]
    prediction_classes = [classes[integer_label] for integer_label in predictions]

    # TODO: Plot images with class names
    # print("images_class_split: {}".format(images_class_split))
    print("images: {}".format(images.shape))
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