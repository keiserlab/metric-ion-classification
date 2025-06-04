import torch
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import f1_score, auc, roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

def get_embeddings(dl, model, return_labels = True, device = 'cuda'):
    embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dl:
            features = features.to(device)
            labels = labels.to(device)
            embeddings.append(model(features).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        
    embeddings = np.concatenate(embeddings)
    all_labels = np.concatenate(all_labels)
    
    return embeddings, all_labels if return_labels else embeddings

def svc_prob(train_embeddings, test_embeddings, train_labels, test_labels, kernel = "linear",
                    return_train = False, return_svc = False, random_state = 27, svc = None):
    if svc is None:
        svc = SVC(kernel = kernel, probability = True, random_state = random_state)
        svc.fit(train_embeddings, train_labels)
    test_probas = svc.predict_proba(test_embeddings)
    if return_train:
        train_probas = svc.predict_proba(train_embeddings)
        return test_labels, test_probas, train_probas
    if return_svc:
        train_probas = svc.predict_proba(train_embeddings)
        return test_labels, test_probas, train_probas, svc
    return test_labels, test_probas

def get_metrics(data = None,
                model = None,
                dataloaders = None,
                metrics = ['auc', 'prc', 'f1', 'acc'],
                kernel = 'linear',
                device = 'cpu',
                avg_type = 'macro',
                embed = True):
    return_metrics = {}
     
    if embed:
        all_embeddings = []
        all_labels = []
        for dataloader in dataloaders:
            embeddings, labels = get_embeddings(dataloader, model, return_labels = True, device = device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
     
        test_labels, test_probas = svc_prob(*all_embeddings, *all_labels, kernel = kernel)
    else:
        test_labels, test_probas = data
        
    test_preds = np.argmax(test_probas, axis = 1)     
    
    if 'acc' in metrics:
        return_metrics['acc'] = sum(test_labels == test_preds)/len(test_labels)
    
    if 'f1' in metrics:
        return_metrics['f1'] = f1_score(test_labels, test_preds, average = 'macro')
    
    roc_info, prc_info = {}, {}
    test_labels_binary = label_binarize(test_labels, classes = list(range(test_probas.shape[1])))
    
    if 'auc' in metrics:
        fpr, tpr, thresh = roc_curve(test_labels_binary.ravel(), test_probas.ravel())
        return_metrics['auc'] = auc(fpr, tpr)
        
    if 'prc' in metrics:
        precision, recall, thresh = precision_recall_curve(test_labels_binary.ravel(), test_probas.ravel())
        return_metrics['prc'] = auc(recall, precision)

    return return_metrics

    
