'''
Name: Alahi, Sk Nasimul (Please write names in <Last Name, First Name> format)

Collaborators: Wong, Alex (Please write names in <Last Name, First Name> format)

Collaboration details: The collaborator provided the skeleton and almost all the
functions required for this program. Also, discussed with him about saving the
output file and overall performance of the program. 

Submission:

TODO: Please submit your <last_name>_<first_name>_exercise10.py along with your checkpoint.pth

Summary:

TODO: Explain your design for your neural network e.g.
How many layers, neurons did you use? What kind of activation function did you use?
Please give reasoning of why you chose a certain number of neurons for some layers.

Ans: As per submission rule, I used Pytorch framework with ReLU activation and
Stochastic Gradient Descent optimization and built a Neural Network. There are 3
fully connected layers with 1024 neurons. The high number of neurons was actually
required for the program to minimize loss and increase accuracy.

TODO: Report your hyper-parameters.
Batch Size: 4
Epoch: 10
Learning rate: 1e-3
Learning rate decay: 0.50
Learning rate decay period: 1
Momentum: 0.90
Lmabda weight decay: 0.0


TODO: Report your scores here. Mean accuracy should exceed 54%
Epoch=1  Loss: 1.859
Epoch=2  Loss: 1.608
Epoch=3  Loss: 1.491
Epoch=4  Loss: 1.421
Epoch=5  Loss: 1.380
Epoch=6  Loss: 1.355
Epoch=7  Loss: 1.341
Epoch=8  Loss: 1.333
Epoch=9  Loss: 1.314
Epoch=10  Loss: 1.303
Mean accuracy over 10000 images: 54 %

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
    type=int, default=10, help='Number of times to iterate through dataset')
parser.add_argument('--learning_rate',
    type=float, default=1e-3, help='Base learning rate (alpha)')
parser.add_argument('--learning_rate_decay',
    type=float, default=0.50, help='Decay rate for learning rate')
parser.add_argument('--learning_rate_decay_period',
    type=float, default=1, help='Period before decaying learning rate')
parser.add_argument('--momentum',
    type=float, default=0.90, help='Momentum discount rate (beta)')
parser.add_argument('--lambda_weight_decay',
    type=float, default=0.0, help='Lambda used for weight decay')


args = parser.parse_args()


class NeuralNetwork(torch.nn.Module):
    '''
    Neural network class of fully connected layers

    Args:
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
    '''

    def __init__(self, n_input_feature, n_output):
        super(NeuralNetwork, self).__init__()

        # TODO: Design your neural network using fully connected layers
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        
        self.fully_connected1 = torch.nn.Linear(n_input_feature, 1024)
        self.fully_connected2 = torch.nn.Linear(1024, 1024)
        self.fully_connected3 = torch.nn.Linear(1024, 1024)
        self.output = torch.nn.Linear(1024, n_output)
        

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
        
        x1 = self.fully_connected1(x)
        theta_x1 = torch.nn.functional.relu(x1)
        
        x2 = self.fully_connected2(theta_x1)
        theta_x2 = torch.nn.functional.relu(x2)
        
        x3 = self.fully_connected3(theta_x2)
        theta_x3 = torch.nn.functional.relu(x3)
        
        output = self.output(theta_x3)

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

    # TODO: Define cross entropy loss
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_decay * param_group['lr']

        for batch, (images, labels) in enumerate(dataloader):

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            n_dim = np.prod(images.shape[1:])
            images = images.view(-1, n_dim)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # TODO: Compute loss function
            loss = loss_func(outputs, labels)

            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
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

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            shape = images.shape
            n_dim = np.prod(shape[1:])
            images = images.view(-1, n_dim)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Take the argmax over the outputs
            _, predictions = torch.max(outputs, dim=1)

            # Accumulate number of samples
            n_sample = n_sample + labels.shape[0]

            # TODO: Check if our prediction is correct
            n_correct = n_correct + torch.sum(predictions == labels).item()

    # TODO: Compute mean accuracy
    mean_accuracy = 100.0 * n_correct / n_sample

    print('Mean accuracy over %d images: %d %%' % (n_sample, mean_accuracy))

    # TODO: Convert the last batch of images back to original shape
    images = images.view(shape[0], shape[1], shape[2], shape[3])
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # TODO: Map the last batch of predictions to their corresponding class labels

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

        index = i - 1
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
        torchvision.transforms.ToTensor(),
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

    # Number of input features: 3 (channel) by 32 (height) by 32 (width)
    n_input_feature = 3 * 32 * 32

    # CIFAR10 has 10 classes
    n_class = 10

    # TODO: Define network
    net = NeuralNetwork(
        n_input_feature=n_input_feature,
        n_output=n_class)
    
    # TODO: Setup learning rate SGD optimizer and step function scheduler
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.lambda_weight_decay,
        momentum=args.momentum)

    if args.train_network:
        # TODO: Set network to training mode
        net.train()

        # TODO: Train network and save into checkpoint
        net = train(
            net=net,
            dataloader=dataloader_train,
            n_epoch=args.n_epoch,
            optimizer=optimizer,
            learning_rate_decay=args.learning_rate_decay,
            learning_rate_decay_period=args.learning_rate_decay_period)

        torch.save({ 'state_dict' : net.state_dict()}, "./checkpoint.pth")
    else:
        # TODO: Load network from checkpoint
        checkpoint = torch.load("./checkpoint.pth")
        net.load_state_dict(checkpoint['state_dict'])

    # TODO: Set network to evaluation mode
    net.eval()

    # TODO: Evaluate network on testing set
    evaluate(
        net=net,
        dataloader=dataloader_test,
        classes=classes)