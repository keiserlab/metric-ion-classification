from mic.analysis.analysis import get_embeddings, svc_prob

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve

import numpy as np
import umap

import matplotlib.pyplot as plt

def plot_latent(train_dl, test_dl, model, model_config, device, label_names, accuracy_calc = 'scv', suffix = "", 
                plot = True, trial_name = "", color_dict = None): 
    print("model", model)

    if len(trial_name) > 0:
        trial_name += ": "
    title_string = trial_name + '_'.join([f'{key}-{val:.3e}' if type(val) == float else f'{key}-{val}' \
                             for key, val in model_config.items() ]) + suffix
    
    all_embeddings = []
    all_labels = []
    for dataloader in [train_dl, test_dl]:
        if model is not None:
            embeddings, labels = get_embeddings(dataloader, model, return_labels = True, device = device)
        else:
            embeddings = np.concatenate([batch[0].detach().cpu().numpy() for batch in dataloader])
            labels = np.concatenate([batch[1].detach().cpu().numpy() for batch in dataloader])
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    len_train = all_embeddings[0].shape[0]
    
    if accuracy_calc == 'knn':
        acc = kNN_accuracy(*all_embeddings, *all_labels) #kNN accuracy calculations
    else:
        #scv with ROC plot
        roc_info = {}
        prc_info = {}
        test_labels, test_probas = svc_prob(*all_embeddings, *all_labels)
        acc = sum([test_labels[i] == np.argmax(test_probas[i, :]) \
                   for i in range(test_probas.shape[0])])/test_probas.shape[0]
        if test_probas.shape[1] == 2: # binary case
            fpr, tpr, thresh = roc_curve(test_labels, test_probas[:, 1]) # assume 2nd class is "positive"
            label_auc = auc(fpr, tpr)
            roc_info[label_names[1]] = [fpr, tpr, thresh, label_auc]
        else:
            # binarize labels
            test_labels_binary = label_binarize(test_labels, classes = list(range(test_probas.shape[1])))
            for label_idx, label_name in enumerate(label_names):
                # roc
                fpr, tpr, thresh = roc_curve(test_labels_binary[:,label_idx], test_probas[:, label_idx]) # assume 2nd class is "positive"
                label_auc = auc(fpr, tpr)
                roc_info[label_name] = [fpr, tpr, thresh, label_auc]
                
                # prc
                precision, recall, thresh = precision_recall_curve(test_labels_binary[:,label_idx], test_probas[:, label_idx]) # assume 2nd class is "positive"
                label_auc = auc(recall, precision)
                prc_info[label_name] = [recall, precision, thresh, label_auc]
                
            # Compute micro-average ROC curve and ROC area
            fpr, tpr, thresh = roc_curve(test_labels_binary.ravel(), test_probas.ravel())
            micro_auc = auc(fpr, tpr)
            roc_info['micro'] = [fpr, tpr, thresh, micro_auc]
                                                      
            precision, recall, thresh = precision_recall_curve(test_labels_binary.ravel(), test_probas.ravel())
            micro_auc = auc(recall, precision)
            prc_info['micro'] = [recall, precision, thresh, micro_auc]
            
    unique, counts = np.unique(test_labels, return_counts=True)
    freq_info = dict((label_names[int(u)], float(c)/test_labels.shape[0]) for u, c in zip(unique, counts))
    freq_info['micro'] = sum(freq_info.values())/len(label_names)    
        
    # umap embed to plot, if needed
    if all_embeddings[0].shape[1] > 2:
        umap_2d = umap.UMAP(n_components=2, init='random', random_state=27)
        proj_2d = umap_2d.fit_transform(np.concatenate(all_embeddings))
    else:
        proj_2d = np.concatenate(all_embeddings)
    
    train_embeddings = proj_2d[:len_train, :]
    train_labels = all_labels[0]

    test_embeddings = proj_2d[len_train:, :]
    test_labels = all_labels[1]

    if accuracy_calc == 'knn':
        ncols = 2
        figwidth = 20
    else:
        ncols = 4
        figwidth = 40
    if plot:
        fig, ax = plt.subplots(nrows = 1, ncols = ncols, figsize = (figwidth, 10))
        for i, (name, embeddings, labels) in enumerate(zip(['Training', 'Testing'], 
                                                        [train_embeddings, test_embeddings],
                                                        [train_labels, test_labels])):
            for j, ion in enumerate(label_names):
                idxs = np.where(labels == j)
                ax[i].scatter(embeddings[idxs, 0], embeddings[idxs, 1], label = ion, color = color_dict[ion])
                if i > 0 and accuracy_calc == 'knn':
                    ax[i].set_title(f"{name} - kNN Accuracy (k = 10): {acc:.3f}")
                else:
                    ax[i].set_title(f"{name}")
                ax[i].legend()

        if accuracy_calc == 'scv': # do we need to plot ROC curve
            for ion_name in roc_info:
                fpr, tpr, thresh, label_auc = roc_info[ion_name]
                ax[2].plot(fpr, tpr, lw = 2, label = f'{ion_name}: {label_auc:.3f}', color = color_dict[ion_name])
            ax[2].set_xlabel("False Positive Rate")
            ax[2].set_ylabel("True Positive Rate")
            ax[2].set_title(f"Average AUROC: {roc_info['micro'][3]:.3f}")
            ax[2].legend(loc="lower right")
            #plot random split
            ax[2].plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle="--")

            for ion_name in prc_info:
                recall, precision, thresh, label_auc = prc_info[ion_name]
                ax[3].plot(recall, precision, lw = 2, label = f'{ion_name}: {label_auc:.3f}', color = color_dict[ion_name])
            ax[3].set_xlabel("Recall")
            ax[3].set_ylabel("Precision")
            ax[3].set_title(f"Average AUPRC: {prc_info['micro'][3]:.3f}, (baseline: {freq_info['micro']:.3f})")
            ax[3].legend(loc="upper right")
            #plot random split
        # update - plot precision/recall curves
        fig.suptitle(f"{title_string}\nAccuracy: {acc:.3f}")
    
    return acc, roc_info['micro'][3], prc_info['micro'][3]
