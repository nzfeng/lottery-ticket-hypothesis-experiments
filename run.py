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
# codebase. They used gradient descent as optimizer, cross-entropy loss, reLU.

# Parameters
n_layer1 = 300 # dimensions of layer 1
n_layer2 = 100 # dimensions of layer 2
n_out = 10 # dimensions of final layer (must be 10)
n_input = 784 # size of image vector


def build_model():

	layers = []
	fcs = []

	w = nn.Linear(n_layer1, n_layer2, bias=True)
	layers.append(('fc{}'.format(1), w))
	fcs.append(w)
	layers.append(('relu{}'.format(1), nn.ReLU()))

	w = nn.Linear(n_layer2, n_out, bias=True)
	layers.append(('fc{}'.format(2), w))
	fcs.append(w)

	model = nn.Sequential(OrderedDict(layers))

	# Return model, as well as list of fully connected hidden layers to do pruning on later.
	return model, fcs


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

	print('Accuracy: %0.5f' %(n_correct / n_total))


def train_epoch(model, train_dataset, optimizer, criterion, layers):
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

	# Output the current loss.
	print('Loss %0.5f' %loss)



def train(model, layers, n_iterations, n_epochs, optimizer, criterion, train_dataset, test_dataset):
	'''
	Train for the given number of iterations/epochs.
	'''
	for i in range(n_iterations):
		print('Iteration %d' %(i+1))
		for e in range(n_epochs):
			# Train for one epoch.
			print('Epoch %d' %(e + 1))
			train_epoch(model, train_dataset, optimizer, criterion, layers)
			# Output the current accuracy.
			test(model, test_dataset)

	# Save trained model.
	p = [(fc.weight.data.numpy(), fc.bias.data.numpy()) for fc in fcs]
	with open(args.save_file, 'wb') as f:
		pickle.dump(p, f)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--iterations', '-i', type=int, default=5, help='Number of training iterations')
	parser.add_argument('--epochs', '-e', type=int, default = 1, help='Number of training epochs')
	parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size')
	parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
	parser.add_argument('--save_file', '-f', type=str)
	args = parser.parse_args()

	# Get the data.
	train_data = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
	test_data = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
	# Transform into a Python iterable.
	train_dataset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
	test_dataset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

	# Build the model.
	model, layers = build_model()
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss(reduction='mean')

	# Train the model!
	train(model, layers, args.iterations, args.epochs, optimizer, criterion, train_dataset, test_dataset)


if __name__ == '__main__':
	main()