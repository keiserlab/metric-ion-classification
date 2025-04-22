import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import os

import json
import joblib

import pandas as pd

# torch classes for training
class FingerprintDataset(Dataset):
    """
    Simple fingerprint dataset
    Takes in a dataframe, returns single fingerprint and label
    """

    def __init__(self, df):
        #set our dataframe
        self.df = df
    
    def __getitem__(self, idx):
        # get fingerprint and label at index idx
        vals = self.df.iloc[idx]
        features = vals.loc[range(4096)].values # fingerprint
        label = vals.loc['label']
        return features.astype(np.float32), label.astype(np.float32)
    
    def __len__(self):
        # how many examples do we have in our dataset?
        return self.df.shape[0]

    
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, dropout_p = 0.1, final_relu=False):
        super().__init__()
        layer_list = []
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
                layer_list.append(nn.Dropout(p = dropout_p))
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)
    
class MIC():
    def __init__(self, fp_type = 'prune-eifp', extended_labels = False, hetatm = False, device = 'cpu'):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.device = device
        available_fp_types = ['prune-eifp']
        if fp_type not in available_fp_types:
            raise ValueError(f'Invalid fp_type {fp_type} requested, please select from {available_fp_types}')
        if extended_labels and hetatm:
            raise ValueError(f'HETATM model only available for prevalent ion dataset.')        

        if extended_labels:
            fp_type = fp_type + '-extended'
            self.labels = np.array(['HOH', 'MG', 'NA', 'ZN', 'CA', 'CL', 'K', 'MN', 'IOD', 'FE', 'BR', 'Null'])
        else:
            self.labels = np.array(['HOH', 'MG', 'NA', 'ZN', 'CA', 'CL', 'Null'])

        if hetatm:
            self.labels = np.append(self.labels, ['HETATM'])
            fp_type = fp_type + '-hetatm'
        
        self._load_model(os.path.join(dir_path, f'trained_models/{fp_type}'), device)
 
    def predict(self, x, entries, return_proba = True, return_confidence = True):
        """
        x - LUNA fingerprints in np array format
        """
        pd.options.display.float_format = '{:.4f}'.format

        embeddings = self.model(torch.from_numpy(x).float().to(self.device)).detach().cpu().numpy()
        predictions = self.svc.predict_proba(embeddings)
        results = pd.DataFrame(predictions, columns=self.labels, index=entries)
        results['prediction'] = self.labels[predictions.argmax(axis=1)]
        results['confidence'] = np.max(predictions, axis=1)

        final_cols = ['prediction']
        if return_confidence: final_cols += ['confidence']
        if return_proba: final_cols += list(self.labels) 
                
        results = results[final_cols].round(4)
        return results
       
    def _load_model(self, model_path, device):
        with open(f'{model_path}.config') as f:
            self.model_config = json.load(f)
        self.model = MLP(self.model_config['layers'], dropout_p = self.model_config['dropout_p'])
        if device=='cpu':
            self.model.load_state_dict(torch.load(f'{model_path}.pt', map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(f'{model_path}.pt'))
        self.model.eval()
        self.model.to(device)
        
        self.svc = joblib.load(f'{model_path}.svc')
