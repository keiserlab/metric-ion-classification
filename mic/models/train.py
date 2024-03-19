import os

import torch
from torch.utils.tensorboard import SummaryWriter

import pytorch_metric_learning
from pytorch_metric_learning import losses, miners, samplers

from ..models.MIC import MLP
from ..analysis.analysis import get_embeddings, svc_prob, get_metrics

import json
import numpy as np

def train_model(iteration, train_loader, val_loader, device, settings, prefix = '', 
                n_epochs = 1000, verbose = False, logdir = '.'):

    # create model according to selected params    
    model = MLP(settings['layers'], settings['dropout_p']).to(device)
    model.to(device)
    
    # optimizer
    optim = torch.optim.Adam(
        model.parameters(), 
        lr=settings['learning_rate'], 
        weight_decay=settings['weight_decay'])
    
    # miner and loss function
    loss_func = losses.TripletMarginLoss(margin = settings['loss_margin'])
    miner = miners.TripletMarginMiner(margin = settings['miner_margin'])
    
    # logging
    logdir = f'{logdir}/{prefix}/{iteration}' 
    logger = SummaryWriter(log_dir = logdir)
    
    #dump settings to file
    with open(f'{logdir}/config.txt', 'w+') as f:
        json.dump(settings, f)
        
    train_losses = []
    val_losses = []
    aucs = []
    f1s = []
    best_f1 = float("-inf") 
    best_auc = float("-inf") 
    for epoch in range(n_epochs):
        epoch_losses = []
        train_embeddings = []
        train_labels = []
        for i, (features, labels) in enumerate(train_loader):
            optim.zero_grad()
            embeddings = model(features.to(device))
            train_embeddings.append(embeddings.detach().cpu().numpy())
            train_labels.append(labels)
            triplets = miner(embeddings, labels.to(device))
            loss = loss_func(embeddings, labels.to(device), triplets)
            epoch_losses.append(loss.item())
            loss.backward()
            optim.step()
            
        train_embeddings = np.concatenate(train_embeddings)
        train_labels = np.concatenate(train_labels)
        
        train_losses.append(np.mean(epoch_losses))
        logger.add_scalar('loss/train', np.mean(epoch_losses), epoch)
        
        with torch.no_grad():
            val_epoch_losses = []
            val_embeddings = []
            val_labels = []
            for features, labels in val_loader:
                embeddings = model(features.to(device))
                val_embeddings.append(embeddings.detach().cpu().numpy())
                val_labels.append(labels)
                hard_pairs = miner(embeddings, labels.to(device))
                loss = loss_func(embeddings, labels.to(device), hard_pairs)
                val_epoch_losses.append(loss.item())
                
            val_epoch_loss = np.mean(val_epoch_losses)
            logger.add_scalar('loss/val', np.mean(val_epoch_losses), epoch)
            val_losses.append(val_epoch_loss)
        
        val_embeddings = np.concatenate(val_embeddings)
        val_labels = np.concatenate(val_labels)
            
        # Calculate auROC and f1 score
        val_labels, val_probas = svc_prob(train_embeddings, val_embeddings, train_labels, val_labels)
        metrics = get_metrics(data = [val_labels, val_probas], embed = False)
        
        logger.add_scalar('metric/auc', metrics['auc'], epoch)
        logger.add_scalar('metric/f1', metrics['f1'], epoch)
        
        aucs.append(metrics['auc'])
        f1s.append(metrics['f1'])
        
        torch.save({"epoch": epoch+1,
                    "state_dict": model.state_dict(),
                    "optim_dict": optim.state_dict()},
                    os.path.join(logdir, 'latest_checkpoint.pt'))
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_auc = metrics['auc']

            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optim_dict': optim.state_dict()},
                        os.path.join(logdir, 'best_checkpoint.pt'))
            
            with open(os.path.join(logdir, 'best_f1.txt'), 'w+') as f:
                f.write(f"Validation: {val_losses[-1]}\n")
                f.write(f"micro AUROC: {metrics['auc']}\n")
                f.write(f"f1: {best_f1}\n")
            
            patience_count = 0 #reset
        else: #validation didn't drop - increase our count
            patience_count += 1
            
        if patience_count > settings['patience']:
            if verbose:
                print(f"Stopping early at epoch {epoch}")
            return best_f1, best_auc
   
    return best_f1, best_auc
