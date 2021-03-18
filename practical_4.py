#!/usr/bin/env python

import torch
from torch import nn
from torch.nn import functional as F

import practical_prologue as prologue

import matplotlib.pyplot as plt 


class Net(nn.Module):
    def __init__(self, n=200):
        # hard coding dims 
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, n)
        self.fc2 = nn.Linear(n, 10)

    def forward(self, x):
        # CONV LAYER 1
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))

        # CONV LAYER 2
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))

        # FC 1
        x = F.relu(self.fc1(x.view(-1, 256)))

        # FC2
        x = self.fc2(x)
        return x


class Net2(nn.Module): 
    def __init__(self, n=200):
        super().__init__()
        # EXPAND WITH CONV 
        # THEN DECREASE WITH LINEAR 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc1 = nn.Linear(256, n)
        self.fc2 = nn.Linear(n, 10)

    def forward(self, x): 
        print('dim x', x.shape)

        # CONV LAYER 1
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        # pooling just divides w, h by kernel size (provided stride=kernelsize)
        print('dim x after relu pool first conv', x.shape)

        # CONV LAYER 2 
        # TODO weirdly this only works when stride < kernel 
        # otherwise it is reshaped into the wrong dimensions for the fc layers 
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2))
        print('dim x ater second conv', x.shape)

        # CONV LAYER 3 
        x = F.relu(self.conv3(x))
        # Needs to be 1000 x 256 
        print('dim x after third conv', x.shape)
    
        # FC1 
        rs = x.view(-1,256)
        print('rs shape', rs.shape)

        x = F.relu(self.fc1(rs))
        print('dim x after first connected layer', x.shape)

        # FC2
        x = self.fc2(x)
        print('dim x after second connected layer', x.shape)
        return x


def train_model(model, criterion, 
                train_input, train_target, mini_batch_size, 
                eta=1e-1, nb_epochs=100, verbose=True):
    for e in range(nb_epochs):
        acc_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

        if verbose:
            print(e, acc_loss)

    return model  


def compute_nb_errors(model, test_input, test_target, mini_batch_size):
    # need model output 
    predictions = model.forward(test_input) 
    prediction_labels = torch.argmax(predictions, dim=1)

    true_labels = torch.argmax(test_target, dim=1) 

    # return % of incorrect predictions 
    incorrect = torch.count_nonzero(true_labels - prediction_labels)
    score = 100 * incorrect / test_target.size(0) 
    return score


def get_train_test_error(model,
                         train_input, train_target, 
                         test_input, test_target, 
                         mini_batch_size): 
    train_score = compute_nb_errors(model, train_input, train_target, mini_batch_size)
    print('train score = {} on {} training points'.format(train_score, train_input.size(0)))

    test_score = compute_nb_errors(model, test_input, test_target, mini_batch_size) 
    print('test score = {} on {} training points'.format(test_score, train_input.size(0)))

    return train_score, test_score 


def get_error_hidden_units(train_input, train_target, 
        test_input, test_target, 
        eta=1e-1, mini_batch_size=100,
        nb_epochs=25,
        n_hidden=[10,50,200,500,1000]):
    train_scores = []
    test_scores = []

    for n in n_hidden:
        print('{} hidden units'.format(n))
        model, criterion = Net(n=n), nn.MSELoss()
        model = train_model(model, criterion, 
                            train_input, train_target, mini_batch_size,
                            eta=eta, nb_epochs=nb_epochs, verbose=False)
        tr_score, te_score = get_train_test_error(model, train_input, train_target, test_input, 
                                                test_target, mini_batch_size)
        train_scores.append(tr_score)
        test_scores.append(te_score)

    fig, ax = plt.subplots()
    ax.plot(n_hidden, train_scores, label="train error")
    ax.plot(n_hidden, test_scores, label="test error")
    ax.set_xlabel("number hidden units")
    ax.set_ylabel("winner take all error")
    plt.legend()
    plt.savefig("convergence_hidden_units.png", dpi=300)
    return 


def train_net_1_iters(
        train_input, train_target, test_input, test_target,
        eta, mini_batch_size, nb_epochs, n
        ):
    model, criterion = Net(n=n), nn.MSELoss()
    # try Net1 
    for i in range(10):
        print('iter',i)
        print('training Net 1')
        model = train_model(model, criterion, 
                            train_input, train_target, mini_batch_size, 
                            eta=eta, nb_epochs=nb_epochs, verbose=False)
        tr_score, te_score = get_train_test_error(model, train_input, train_target, 
                                                test_input, test_target, mini_batch_size)
    return model  


def train_net_2(train_input, train_target, 
                test_input, test_target,
                mini_batch_size, eta, nb_epochs, n):
    model, criterion = Net2(n=n), nn.MSELoss()
    print('training Net 2')

    model = train_model(model, criterion,
                        train_input, train_target, mini_batch_size, 
                        eta=eta, nb_epochs=nb_epochs, verbose=False)

    tr_score, te_score = get_train_test_error(model, train_input, train_target,
                                              test_input, test_target, mini_batch_size)
    return model 


if __name__ == "__main__":
    # SMALL DATA SET 
    train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

    eta = 1e-1
    mini_batch_size = 1000
    nb_epochs = 1
    n = 200

    # NET 1 wrt hidden unit size
    #get_error_hidden_units(train_input, train_target, test_input, test_target) 
    #train_net_1_iters(train_input, train_target, test_input, test_target, eta, 
    #                  mini_batch_size, nb_epochs, n)

    # NET 2 
    train_net_2(train_input, train_target, test_input, test_target, mini_batch_size, 
                eta, nb_epochs, n)

    # STRIDE connects pixels neighbours 
    # DILATION connects pixels far apart - background colour  
    # Remember dimensions are 
    # init N x C X W x L 
    # conv N x D x W - w + 1 x L - l + 1 
    # if using strides divide W and L by stride size 
    # dilation etc more complicated 
