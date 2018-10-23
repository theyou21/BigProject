from __future__ import print_function, division
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

class Solver(object):
    
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        
        # Unpack keyword arguments
        self.lr = kwargs.pop('lr', 1.0e-3)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 50)
        self.num_epochs = kwargs.pop('num_epochs', 50)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.dtype = kwargs.pop('dtype', '32f')
        self.optimizer =kwargs.pop('optimizer', optim.Adam(model.parameters(), 
                                    lr=self.lr))
        

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)


        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.best_model = None

        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        if self.dtype == '16bit':
            realtype = torch.HalfTensor
        elif self.dtype == '64bit':
            realtype = torch.DoubleTensor
        else:
            realtype = torch.FloatTensor

        self.loss = nn.CrossEntropyLoss().type(realtype)

        self.X_train = Variable(torch.Tensor(self.X_train)).type(realtype)
        self.y_train = Variable(torch.Tensor(self.y_train)).type(torch.LongTensor)
        self.X_val = Variable(torch.Tensor(self.X_val)).type(realtype)
        self.y_val = Variable(torch.Tensor(self.y_val)).type(torch.LongTensor)
        

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask,]
        y_batch = self.y_train[batch_mask,]
        
        # calculate training loss
        y_batch_pred = self.model(X_batch)
        loss = self.loss(y_batch_pred, y_batch)
        self.loss_history.append(loss.data.numpy())

        # backpropagation
        self.model.zero_grad()
        loss.backward()

        # update parameters
        self.optimizer.step()
        
    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:

                # calculate training accuracy
                y_train_pred = self.model(self.X_train)
                _, y_pred = torch.max(y_train_pred,1)
                train_acc = np.mean(y_pred.data.numpy() == 
                            self.y_train.data.numpy())

                # calculate validation accuracy
                y_val_pred = self.model(self.X_val)
                _, y_pred = torch.max(y_val_pred,1)
                val_acc = np.mean(y_pred.data.numpy() == 
                            self.y_val.data.numpy())
                
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    self.best_model = self.model

        # At the end of training swap the best params into the model
        self.model = self.best_model