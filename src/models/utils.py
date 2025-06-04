from ..models.structures import MLP

import os
import json
import string, random
import torch
from torch.utils.data import WeightedRandomSampler


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# setup model according to existing config file
def load_model(model_path, device, model_config = None):
    if model_config is None:
        with open(os.path.join(model_path, 'config.txt')) as f:
            model_config = json.load(f)
    
    model = MLP(model_config['layers'], dropout_p = model_config['dropout_p'])
    
    checkpoint = torch.load(os.path.join(model_path, 'best_checkpoint.pt'),
            map_location=torch.device('cpu') )
        
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    model.to(device)
    
    return model, model_config

def get_weighted_sampler(df):
    labelcount_df = df.groupby('site_id').count().label
    weight_dict = dict([(labelcount_df.index[i], 1/labelcount_df[i]) for i in range(labelcount_df.shape[0])])
    trainweights = [weight_dict[ion] for ion in df.site_id]
    weightedsampler = WeightedRandomSampler(trainweights, 
                            num_samples = len(trainweights), replacement = True)
    return weightedsampler
