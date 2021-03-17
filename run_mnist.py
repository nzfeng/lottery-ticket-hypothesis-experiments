import numpy as np
import argparse
from collections import OrderedDict
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Train and test Lenet 300-100 model on MNIST dataset, which was used in the original Lottery Ticket Hypothesis 
# codebase. They used gradient descent as optimizer, cross-entropy loss, reLU activation.

# Parameters
n_input = 784 # size of image vector
n_layer1 = 300 # dimensions of layer 1
n_layer2 = 100 # dimensions of layer 2
n_out = 10 # dimensions of final layer (must be 10)
epsilon = 1e-8

def replace_model_layer(model, layer_name, new_layer_weight):
    try:
        setattr(model, layer_name, new_layer_weight)
    except:
        print("abort")

#Adapted from salesforce research
class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()


    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.Parameter(torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training))
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

class Lenet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Lenet, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_layer1, bias=True)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(n_layer1, n_layer2, bias=True)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(n_layer2, output_dim, bias=True)
        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def build_model():
    net = Lenet(n_input, n_out)
    return net, net.layers 


def test(model, test_dataset):
    '''
    Return the accuracy of the trained model on the test dataset.
    '''
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for (inputs, labels) in test_dataset:
            inputs = inputs.view(-1, 784)
            n_total += len(inputs)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            n_correct += (predictions == labels).sum().item()

    print('Test Accuracy: %0.5f' %(n_correct / n_total))


def train_epoch(model, masks, train_dataset, optimizer, criterion, layers):
    '''
    Train model for one epoch.
    '''

    total_loss = 0
    model.train()
    for (inputs, labels) in train_dataset:
        inputs = inputs.view(-1, 784)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss

        for l in range(len(layers)):
            layers[l].weight.data *= masks[l]

    # Output the current loss.
    print('\tLoss %0.5f' %loss)



def compute_density(layers):
    '''
    Compute the overall density of the model.
    '''
    n_weights = 0
    n_nonzero = 0
    for l in range(len(layers)):
        (rows, cols) = layers[l].weight.data.shape
        n_weights += rows * cols
        n_nonzero += (abs(layers[l].weight.data) > epsilon).sum().sum().item()

    return n_nonzero / n_weights



def train(model, layers, masks, n_iterations, n_epochs, optimizer, criterion, train_dataset, test_dataset):
    '''
    Train for the given number of iterations/epochs.
    '''
    for i in range(n_iterations):
        print('\nIteration %d' %(i+1))
        for e in range(n_epochs):
            # Train for one epoch.
            print('\tEpoch %d' %(e + 1))
            train_epoch(model, masks, train_dataset, optimizer, criterion, layers)
            # Output the current accuracy.
            test(model, test_dataset)
            # Output the current sparsity.
            print('Overall density: %0.5f' %compute_density(layers))



def save_model(layers, save_file):
    '''
    Save a trained model ('layers') to the filepath given by save_file.
    '''
    p = [(fc.weight.data.numpy(), fc.bias.data.numpy()) for fc in layers]
    with open(save_file, 'wb') as f:
        pickle.dump(p, f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-T', type=int, default=2, help='Number of training iterations T')
    parser.add_argument('--runs', '-R', type=int, default=2, help='Number of runs in IMP (pruning steps - 1)')
    parser.add_argument('--rewind_iteration', '-k', type=int, default=1, help='Parameter k in IMP w/ rewinding')
    parser.add_argument('--prune_levels', '-p', type=tuple, default=(0.99,0.99,0.99), help='Pruning level per layer')
    parser.add_argument('--epochs', '-e', type=int, default = 1, help='Number of training epochs per iteration')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Learning rate in SGD')
    parser.add_argument('--save_file', '-f', type=str, default="./models/mnist.pkl", help='Location to save model')
    args = parser.parse_args()

    # IMP parameters
    T = args.iterations
    R = args.runs
    k = args.rewind_iteration
    prune_levels = args.prune_levels

    # Get the data.
    train_data = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    test_data = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    # Transform into a Python iterable.
    train_dataset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Build the model.
    model, layers = build_model()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    masks = [] # the jth entry represents the mask applied to the jth layer
    for l in range(len(layers)):
        masks.append(torch.ones(layers[l].weight.data.shape, dtype=torch.float))
    # Train the model for one run.
    train(model, layers, masks, k, args.epochs, optimizer, criterion, train_dataset, test_dataset)
    weights_k = []
    # Make a note of the weights at iteration k.
    for l in range(len(layers)):
        weights_k.append((layers[l].weight.data, layers[l].bias.data))

    # Train using IMP. See Algorithm 1 in https://arxiv.org/pdf/1903.01611.pdf.
    for round_num in range(R):
        print("Starting new run %d" %round_num)
        # Train for more iterations for T total.
        train(model, layers, masks, T-k, args.epochs, optimizer, criterion, train_dataset, test_dataset)
        # Prune weights (i.e. update masks.)
        for l in range(len(layers)):
            m = masks[l]
            w = layers[l].weight.data.abs() 
            n = (m > 0).sum().sum().item() # number of nonzero weights
            d = int(n * prune_levels[l]) # number of weights to prune
            curr_w = w[m > 0].view(-1) # weights with current mask applied
            r = curr_w.sort().values[d] # get the dth smallest value in the weight matrix
            m[w*m < r] = 0 # set all weights below this value to 0
            masks[l] = m # update mask
            layers[l].weight.data *= m
            
            replace_model_layer(model, str(layers[l]), layers[l].weight.data)
            # model.layers
        print("final test")
        test(model, test_dataset)
        # Reset weights to what they were at iteration k.
        for l in range(len(layers)):
            (layers[l].weight.data, layers[l].bias.data) = weights_k[l]
    # Save the trained model.
    save_model(layers, args.save_file)


if __name__ == '__main__':
    main()