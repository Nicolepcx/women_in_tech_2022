# standard imports
import numpy as np
import os
import sys
import csv
import random
from collections import Counter
import pandas as pd
from datetime import datetime
from importlib import reload

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class AttLSTMVAE(nn.Module):
    def __init__(self, z_dim, T, x_dim, h_dim, model_path, batch_size=1, x_noise_factor=0.1):
        super(AttLSTMVAE, self).__init__()

        self.x_noise_factor = x_noise_factor
        self.z_dim = z_dim
        self.T = T
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.init_hidden()
        self.model_path = model_path

        # Everything going on in the network has to be of size:
        # (batch_size, T, n_features)

        # We encode the data onto the latent space using bi-directional LSTMs
        self.LSTM_encoder = nn.LSTM(
            input_size=self.x_dim,
            hidden_size=self.h_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.fnn_zmu = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )

        self.fnn_zvar = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )

        self.fnn_cmu = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )

        self.fnn_cvar = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.z_dim,
            bias=True
        )

        # The latent code must be decoded
        self.LSTM_decoder = nn.LSTM(
            input_size=self.z_dim*2,
            hidden_size=self.h_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.fnn_xmu = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.x_dim,
            bias=True
        )

        self.fnn_xb = nn.Linear(
            in_features=2*self.h_dim,
            out_features=self.x_dim,
            bias=True
        )

    def init_hidden(self):
        self.hidden_state = torch.zeros(2, self.batch_size, self.h_dim)
        self.cell_state = torch.zeros(2, self.batch_size, self.h_dim)
        if torch.cuda.is_available():
            self.hidden_state = self.hidden_state.to(torch.device(0))
            self.cell_state = self.cell_state.to(torch.device(0))

    def encoder(self, x):
        out_encoded, (hidden_T, _) = self.LSTM_encoder(
            x, (self.hidden_state, self.cell_state))

        # Swap batch dimension to batch_first
        hidden_T = hidden_T.transpose(0, 1)

        # Flatten
        flat_hidden_T = hidden_T.reshape(self.batch_size, 1, 2*self.h_dim)

        # Fully connected layer from LSTM to var and mu
        mu_z = self.fnn_zmu(flat_hidden_T)
        sigma_z = nn.functional.softplus(self.fnn_zvar(flat_hidden_T))

        # Attention mechanismn
        # Calculate the similarity matrix
        #print("out_encoded = ", out_encoded.shape)
        S = torch.matmul(
            out_encoded,
            torch.transpose(out_encoded, 1, 2)
        )

        S = S / np.sqrt((2 * self.h_dim))

        # Use softmax to get the sum of weights to equal 1
        A = nn.functional.softmax(S, dim=2)
        Cdet = torch.matmul(A, out_encoded)

        # Fully connected layer from LSTM to var and mu
        mu_c = self.fnn_cmu(Cdet)
        sigma_c = nn.functional.softplus(self.fnn_cvar(Cdet))

        return mu_z, sigma_z, mu_c, sigma_c, A

    def decoder(self, c, z):
        # Concatenate z and c before giving it as input to the decoder
        z_cat = torch.cat(self.T*[z], dim=1)
        zc_concat = torch.cat((z_cat, c), dim=2)

        # Run through decoder
        out_decoded, _ = self.LSTM_decoder(zc_concat)

        # Pass the decoder outputs through fnn to get LaPlace parameters
        mu_x = self.fnn_xmu(out_decoded)
        b_x = nn.functional.softplus(self.fnn_xb(out_decoded))

        return mu_x, b_x

    def forward(self, x):
        if self.training:
            var_x = self.x_noise_factor * \
                torch.var(x, dim=1)  # (batch_size, x_dim)
            noise = torch.randn(self.T, self.batch_size, self.x_dim)

            # (48, 16, 1) * (16, 1)
            if torch.cuda.is_available():
                noise = noise.to(torch.device(0))

            # (T, batch_size, x_dim)
            x_noise = torch.mul(noise, torch.sqrt(var_x))
            x = x + torch.transpose(x_noise, 0, 1)  # (batch_size, T, x_dim)

        outputs = {}

        mu_z, sigma_z, mu_c, sigma_c, A = self.encoder(x)

        # Don't propagate gradients through randomness
        with torch.no_grad():
            epsilon_z = torch.randn(self.batch_size, 1, self.z_dim)
            epsilon_c = torch.randn(self.batch_size, self.T, self.z_dim)

        if torch.cuda.is_available():
            epsilon_z = epsilon_z.to(torch.device(0))
            epsilon_c = epsilon_c.to(torch.device(0))

        z = mu_z + epsilon_z * sigma_z
        c = mu_c + epsilon_c * sigma_c

        mu_x, b_x = self.decoder(c, z)

        outputs["z"] = z
        outputs["mu_z"] = mu_z
        outputs["sigma_z"] = sigma_z
        outputs["c"] = c
        outputs["mu_c"] = mu_c
        outputs["sigma_c"] = sigma_c
        outputs["mu_x"] = mu_x
        outputs["b_x"] = b_x
        outputs["A"] = A

        return outputs

    def count_parameters(self):
        n_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return n_grad, n_total



  


def similarity_score(net, x, L, ELBO_loss):

    with torch.no_grad():
        # Pass sequence through encoder to get params in q(z) and p(c)
        mu_z, sigma_z, mu_c, sigma_c, _ = net.encoder(x)

        score = 0

        for _ in range(L):

            # Sample a random c and z vector and reparametrize
            epsilon_c = torch.randn(net.batch_size, net.T, net.z_dim)
            epsilon_z = torch.randn(net.batch_size, 1, net.z_dim)

            if torch.cuda.is_available():
                epsilon_c = epsilon_c.to(torch.device(0))
                epsilon_z = epsilon_z.to(torch.device(0))

            c = mu_c + epsilon_c * sigma_c
            z = mu_z + epsilon_z * sigma_z

            # Pass sample through decoder and calculate reconstruction probability
            mu_x, b_x = net.decoder(c, z)
            pdf = ELBO_loss.pdf_distribution(mu_x, b_x)
            score += pdf.log_prob(x)

        # Average over number of iterations
        return score/L

def get_sim_scores(net, dataset, L, ELBO_loss):

    batch_size = min((len(dataset), dataset.T_w*100))

    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    # Re-initialize network
    net.batch_size = batch_size
    net.init_hidden()
    net.eval()
    all_sim_scores = np.zeros((0, dataset.T_w))

    for x, label in loader:
        # Modify batch size and hidden state
        if x.shape[0] != batch_size:
            net.batch_size = x.shape[0]
            net.init_hidden()

        # Cast to gpu if available
        if torch.cuda.is_available():
            x = x.to(torch.device(0))

        temp_sim = tensor2numpy(similarity_score(
            net, x, L, ELBO_loss)).reshape(-1, dataset.T_w)
        all_sim_scores = np.concatenate((all_sim_scores, temp_sim))

    return all_sim_scores

def get_z_values(net, dataset, batch_size):

    # Set batch size to length of dataset
    net.batch_size = batch_size
    net.init_hidden()
    net.eval()

    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    z_all = np.zeros((0, net.z_dim))
    with torch.no_grad():
        for x, label in loader:
            if torch.cuda.is_available():
                x = x.to(torch.device(0))

            # Change batch size in last iteration
            if not x.shape[0] == batch_size:
                net.batch_size = x.shape[0]
                net.init_hidden()

            output = net(x)

            # Extract z
            z = output['z']
            z_all = np.concatenate((z_all, tensor2numpy(z[:, 0])))

    # Restore batch_size
    net.batch_size = batch_size
    net.init_hidden()

    return z_all

def tensor2numpy(x):
    if x.requires_grad:
        x = torch.Tensor.cpu(x).detach().numpy()
    else:
        x = torch.Tensor.cpu(x).numpy()
    return x


def save_model(net, name):
    _drive = model_path
    f_name = name + ".pth"
    f_path = os.path.join(_drive, f_name)
    torch.save(net.state_dict(), f_path)

def load_model(parameters, name):
    _drive = model_path
    f_name = name + ".pth"
    f_path = os.path.join(_drive, f_name)

    model = VRASAM(*parameters)
    model.load_state_dict(torch.load(f_path))
    if torch.cuda.is_available():
        model = model.to(torch.device(0))

    return model

