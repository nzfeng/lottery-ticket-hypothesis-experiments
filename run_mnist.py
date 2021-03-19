import argparse
import pickle
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.nn import Parameter
from collections import OrderedDict
from adjustText import adjust_text

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
        print(layer_name)
        setattr(model, layer_name, new_layer_weight)
    except:
        print("abort")

#Adapted from salesforce research
class LockedDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        x = x.clone()

        mask = x.new_empty(x.size(0), x.size(1), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


#Adapted from salesforce research
def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """
    # print("hehe", weights, dropout)
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        >>> from torchnlp.nn import WeightDrop
        >>> import torch
        >>>
        >>> torch.manual_seed(123)
        <torch._C.Generator object ...
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)
        >>>
        >>> input_ = torch.randn(3, 2)
        >>> hidden_state = torch.randn(3, 2)
        >>> weight_drop_gru(input_, hidden_state)
        tensor(... grad_fn=<AddBackward0>)
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward
        self.module = module


class Lenet(nn.Module):
    def __init__(self, input_dim, output_dim, weight_drop=0, lock_drop=0):
        super(Lenet, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_layer1, bias=False) if not weight_drop else WeightDrop(nn.Linear(input_dim, n_layer1, bias=False), ['weight'], dropout=weight_drop)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(n_layer1, n_layer2, bias=False) if not weight_drop else WeightDrop(nn.Linear(n_layer1, n_layer2, bias=False), ['weight'], dropout=weight_drop)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(n_layer2, output_dim, bias=False) if not weight_drop else WeightDrop(nn.Linear(n_layer2, output_dim, bias=False), ['weight'], dropout=weight_drop)
        self.layers = [self.fc1, self.fc2, self.fc3]

        self.lock_drop_val = lock_drop
        self.lock_drop = LockedDropout(p=lock_drop)
        

    def forward(self, x, is_train=True):
        if is_train and self.lock_drop_val:
            x = self.lock_drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_model(weight_drop=0, lock_drop=0):
    net = Lenet(n_input, n_out, weight_drop=weight_drop, lock_drop=lock_drop)
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
            outputs = model(inputs, is_train=False)
            predictions = torch.argmax(outputs, dim=1)
            n_correct += (predictions == labels).sum().item()

    print('Test Accuracy: %0.5f' %(n_correct / n_total))
    return n_correct / n_total


def train_epoch(model, masks, train_dataset, optimizer, criterion):
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

        if masks:
            for l in range(len(model.layers)):
                model.layers[l].weight.data *= masks[l]
    # Output the current loss.
    print('Loss %0.5f' %loss)


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



def train(model, masks, n_iterations, n_epochs, optimizer, criterion, train_dataset, test_dataset, base_iter=0):
    '''
    Train for the given number of iterations/epochs.
    '''
    for i in range(n_iterations):
        if (base_iter + i + 1) % 5 == 0:
        	print('\nIteration %d' %(base_iter+i+1))
        for e in range(n_epochs):
            train_epoch(model, masks, train_dataset, optimizer, criterion)
            # Output the current sparsity.
            if masks and (base_iter + i + 1) % 5 == 0:
                print('Overall density: %0.5f' %compute_density(model.layers))



def save_model(layers, save_file):
    '''
    Save a trained model ('layers') to the filepath given by save_file.
    '''
    p = [fc.weight.data.numpy() for fc in layers]
    with open(save_file, 'wb') as f:
        pickle.dump(p, f)

def plot_result(accuracy, levels, mode):
    plt.scatter(levels, accuracy)
    texts = []
    for xy in zip(levels, accuracy):
        texts.append(plt.text(xy[0], xy[1], "%.3f, %.3f"%(xy[0],xy[1])))

    adjust_text(texts, x=levels, y=accuracy, autoalign='x',
            only_move={'points':'y', 'text':'y'}, force_points=0.15,
            arrowprops=dict(arrowstyle="-", color='r', lw=0.5))
    plt.xlabel("levels")
    plt.ylabel("Accuracy")
    plt.title(f"{mode} plot")
    plt.savefig(f'./{mode}.png')
    return 


def weight_drop(args, T, train_data, test_data, train_dataset, test_dataset):
    weight_dropout = 0.1
    weight_dropouts = []
    plot_accs = []
    for round_num in range(args.runs):
        print(f"Round: {round_num}, weight drop: {weight_dropout}")
        print("---------------------------------------")
        model, layers = build_model(weight_drop=weight_dropout)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        train(model, None, T, args.epochs, optimizer, criterion, train_dataset, test_dataset)
        test_acc = test(model, test_dataset)

        plot_accs.append(test_acc)
        weight_dropouts.append(weight_dropout)
        if (1.0 - weight_dropout) / 4 < 0.05:
        	break
        weight_dropout = weight_dropout + (1.0 - weight_dropout) / 4

    plot_result(plot_accs, weight_dropouts, "weight_drop")

def lock_drop(args, T, train_data, test_data, train_dataset, test_dataset):
    lock_dropout = 0.1
    lock_dropouts = []
    plot_accs = []
    for round_num in range(args.runs):
        print(f"Round: {round_num}, lock drop: {lock_dropout}")
        print("---------------------------------------")
        model, layers = build_model(lock_drop=lock_dropout)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        train(model, None, T, args.epochs, optimizer, criterion, train_dataset, test_dataset)
        test_acc = test(model, test_dataset)

        plot_accs.append(test_acc)
        lock_dropouts.append(lock_dropout)

        if (1.0 - lock_dropout) / 4 < 0.05:
        	break
        lock_dropout = lock_dropout + (1.0 - lock_dropout) / 4

    plot_result(plot_accs, lock_dropouts, "lock_drop")

def IMP(args, T, R, k, prune_levels, train_data, test_data, train_dataset, test_dataset):
	# Build the model.
    model, layers = build_model()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    
    masks = [] # the jth entry represents the mask applied to the jth layer
    for l in range(len(layers)):
        masks.append(torch.ones(layers[l].weight.data.shape, dtype=torch.float))

    # Train the model to record the weight at the rewinding level
    train(model, masks, k, args.epochs, optimizer, criterion, train_dataset, test_dataset)
    weights_k = []
    # Make a note of the weights at iteration k.

    for l in range(len(layers)):
        weights_k.append(copy.deepcopy(layers[l].weight.data))

    # Continue training to obtain the test accuracy at T
    train(model, masks, T-k, args.epochs, optimizer, criterion, train_dataset, test_dataset, base_iter=k)
    test_acc = test(model, test_dataset)
    plot_accs, plot_prune_levels = [test_acc], [0.0]
    print(f"Starting T* accuracy {test_acc}")

    # Train using IMP. See Algorithm 1 in https://arxiv.org/pdf/1903.01611.pdf.
    for round_num in range(R):
        print(f"Round: {round_num}, rewind_iteration: {k}, prune_level: {prune_levels[0]}")
        print("---------------------------------------")
        for l in range(len(layers)):
            masks[l] = torch.ones(layers[l].weight.data.shape, dtype=torch.float)

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
     
        # Rewind weights to what they were at iteration k and mask them.
        for l in range(len(layers)):
        	model.layers[l].weight.data = copy.deepcopy(weights_k[l])

        # Now train the new model with T - k round to compare the accuracy
        train(model, masks, T-k, args.epochs, optimizer, criterion, train_dataset, test_dataset, base_iter=k)
        test_acc = test(model, test_dataset)
        plot_accs.append(test_acc)
        plot_prune_levels.append(prune_levels[0])

        if (1.0 - prune_levels[0]) / 4 < 0.05:
        	break
        for i, prune_level in enumerate(prune_levels):
        	prune_levels[i] = prune_levels[i] + (1.0 -prune_levels[i]) / 4
    plot_result(plot_accs, plot_prune_levels, "prune")
    save_model(model.layers, args.save_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-T', type=int, default=6, help='Number of training iterations T')
    parser.add_argument('--runs', '-R', type=int, default=10, help='Number of runs in IMP (pruning steps - 1)')
    parser.add_argument('--rewind_iteration', '-k', type=int, default=3, help='Parameter k in IMP w/ rewinding')
    parser.add_argument('--prune_levels', '-p', type=tuple, default=[0.1,0.1,0.1], help='Pruning level per layer')
    parser.add_argument('--epochs', '-e', type=int, default = 1, help='Number of training epochs per iteration')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Learning rate in SGD')
    parser.add_argument('--mode', '-m', type=str, default="prune", help='Learning rate in SGD')
    parser.add_argument('--save_file', '-f', type=str, default="./models/mnist.pkl", help='Location to save model')
    args = parser.parse_args()

    # IMP parameters
    T = args.iterations
    R = args.runs
    k = args.rewind_iteration
    prune_levels = args.prune_levels
    mode = args.mode

    # Get the data.
    train_data = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    test_data = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

    # Transform into a Python iterable.
    train_dataset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    if mode == "prune":
    	IMP(args, T, R, k, prune_levels, train_data, test_data, train_dataset, test_dataset)
    elif mode == "weight_drop":
    	weight_drop(args, T, train_data, test_data, train_dataset, test_dataset)
    elif mode == "lock_drop":
    	lock_drop(args, T, train_data, test_data, train_dataset, test_dataset)
    


if __name__ == '__main__':
    main()