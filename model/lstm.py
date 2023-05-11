import sys
import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor



class CLSTM_cell(nn.Module):
    def __init__(self, shape, input_channels, filter_size, num_features) -> None:
        super(CLSTM_cell, self).__init__()

        self.shape = shape
        self.input_channels = input_channels 
        self.num_features = num_features  
        self.filter_size = filter_size
        self.stride = 1
        self.padding = int((filter_size - 1) / 2)
        
        '''
            input channels: we concatenate the current input with the previous hidden state in order to learn a pattern between them
            output channels: we multiply this by 4 considering we need to learn four sets of filters corresponding to the input gate, forget gate, output gate and cell gate
        '''

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.num_features, out_channels=4 * self.num_features, kernel_size=self.filter_size, stride=self.stride, padding=self.padding)

    def forward(self, input, hidden_states):
        
        hidden_state, memory_cell = hidden_states
        combined = torch.cat((input, hidden_state), dim=1) 
        conv_combined = self.conv(combined) 



        # split the gates
        gates = torch.split(conv_combined, self.num_features, dim=1)
        input_gate, forget_gate, output_gate, cell_gate = gates

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_gate = torch.tanh(cell_gate)

        # add new info to memory cell
        memory_cell = forget_gate * memory_cell + input_gate * cell_gate
        
        # compute new hidden state
        hidden_state = output_gate * torch.tanh(memory_cell) 

        return hidden_state, memory_cell

    def init_hidden(self, batch_size):
        '''
            Initialize all hidden layers to tensors of zeros
        '''
        return (Tensor(torch.zeros(batch_size,self.num_features,self.shape[0],self.shape[1])).cuda(), Tensor(torch.zeros(batch_size,self.num_features,self.shape[0],self.shape[1])).cuda())


class ConvLSTM(nn.Module):
    ''' Parameters:
            1. filter_size: specifying the size of the kernel to convolve with the image
            2. input_channels: the number of channels our input has - 3 for RGB, 1 for Grayscale (in the first layer)
            3. shape: the output shape after the process
            4. num_features: the output number of channels - each channel correspond to a certain feature (equal to the number of filters used)
            5. num_layers: the depth we want to go to - results in higher number of features extracted
      '''

    def __init__(self, shape, filter_size, input_channels, num_features, num_layers) -> None:
        super(ConvLSTM, self).__init__()
        
        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers

        cell_list = []
        cell_list.append(CLSTM_cell(self.shape, self.input_channels, self.filter_size, self.num_features).cuda()) 

        for _ in range(1, self.num_layers): # start from 1 because we already added one layer - that layer's input channel is the number of channels in the input image
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features).cuda())
        
        self.cell_list = nn.ModuleList(cell_list) 

    def forward(self, input, hidden_state):
        '''
            Computes new hidden state for each layer by adding the current timestep's input information
            Concatenates hidden state information to the input
        '''
        next_hidden = [] 

        for id in range(self.num_layers):
            hidden_channel = hidden_state[id] 
            output = [] 
            for timestep in range(input.shape[0]):
                hidden_channel = self.cell_list[id](input[timestep, ...], hidden_channel)
                output.append(hidden_channel[0]) 

            next_hidden.append(hidden_channel)
            input = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return next_hidden, input

    def init_hidden(self, batch_size):
        '''
            Contains initial states of each hidden layer
        '''

        init_states = []
        for layer in range(self.num_layers):
            init_states.append(self.cell_list[layer].init_hidden(batch_size))

        return init_states 