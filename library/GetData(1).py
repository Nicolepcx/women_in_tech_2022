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



class GetData(Dataset):

    def __init__(self, T_w,filepath, subset='train', smooth_window=0):

      datafolder = os.path.normpath(filepath)
      path = os.path.normpath(f"{datafolder}/{subset}/data.txt")
      if subset != 'train' and subset != 'val':
        label_path = os.path.normpath(f"{datafolder}/{subset}/labels.txt")

      self.data = np.genfromtxt(path)
      if smooth_window > 0:
        self.data = self.moving_average(self.data, smooth_window)
     
      self.T_w = T_w
      self.N_seq = len(self.data)//self.T_w
      
            
        
      self.data = self.data[:(self.N_seq*self.T_w)] # Extract wanted number of sequences
      self.labels = ["None"]*self.N_seq # each sequence has one label
      
      if subset != 'train' and subset != 'val' and os.path.isfile(label_path):
        # Extract wanted number of sequences
        
          self.labels = np.genfromtxt(label_path, dtype="|U5")#[1:]
          self.labels = self.labels[:(self.N_seq*self.T_w)]
          # reshape the anom_labels so the number of rows is the N_seq      
          self.anom_labels = self.gen_anom_labels(self.data)
        

      #print(self.data)
      self.X = torch.Tensor(self.data).view(-1, self.T_w)
      self.time_labels = list(range(len(self.X)))

    def moving_average(self, x, w):
      return np.convolve(x, np.ones(w), 'valid') / w

    def get_regular_label(self):
      return 'None'

    def __len__(self):
      return len(self.X)

    def __getitem__(self, idx):
      x = self.X[idx]
      x = x.view(*x.shape, 1)
      return x, self.time_labels[idx]

    def get_window(self, idx):
      x = self.X[idx]
      x = x.view(1, -1, 1)
      return x

    def get_data(self):
      X = self.X.reshape(-1, self.T_w) 
      labels = self.get_label_sequence()

      return X, labels

    # ------------------------------------------------
    # Methods that only work when the labels are known
    #-------------------------------------------------
    def get_outlier_types(self):
      unique_labels = np.unique(self.labels)
      return unique_labels[unique_labels != self.get_regular_label()]
      

    def get_labels_colors(self):
      all_labels = self.get_all_labels()
      colors = ['black', 'darksalmon', 'springgreen', 'skyblue', 'fuchsia', 'orangered']
      col2label = {label:color for label, color in zip(all_labels, colors)}
      
      # make sure all labels are in the dict col2label
      for label in all_labels:
        if label not in col2label:
          raise KeyError(f"label <{label}> was not inserted in the dictionary. ADD more colors")

      return col2label
                

    def gen_anom_labels(self, x):
      #tstart = np.where((x[1:] != x[:-1]) & (x[:-1] == 'None'))[0]+1
      #tend = np.where((x[:-1] != x[1:]) & (x[1:] == 'None'))[0]
      tstart = np.arange(0,((self.N_seq)*self.T_w)+1,self.T_w)
      tend = np.arange((self.T_w-1),((self.N_seq)*self.T_w)+1,self.T_w)
      
      
      #if x[-1] != 'None':
      #  tend = np.append(tend, len(x)-1)
      return list(zip(tstart,tend))


    def get_all_labels(self):
      return list(self.get_outlier_types()) + [self.get_regular_label()]

    def get_label_sequence(self):
      labels = self.labels.reshape(-1, self.T_w)
      labels_seq=[]
      regular_label = self.get_regular_label()
      for seq in labels:
        if all(seq == regular_label):
          labels_seq.append(regular_label)
        else:
          dict_outlier_counts = Counter(seq[seq != regular_label])
          outlier_most_common = max(dict_outlier_counts, key = dict_outlier_counts.get)
          labels_seq.append(outlier_most_common)

      return np.array(labels_seq)

