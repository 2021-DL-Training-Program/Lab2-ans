# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 19:22:43 2021

@author: User
"""

import json
import torch
import os
from torch.utils.data import Dataset   
    
def load(file_name):
    with open(file_name, 'r') as jsonfile:
        content = json.load(jsonfile)
        pairs = []
        for data in content:
            for i in data['input']:
                pairs.append([i, data['target']])
        return pairs
        
if __name__ == '__main__':
    data = load('train.json')
    print(data)
  
    