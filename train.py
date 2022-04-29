import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from tqdm import tqdm

import config as c
from localization import export_gradient_maps
from model import NormFlow, save_model, save_weights
from utils import *


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))


def train(train_loader, test_loader):
    model = NormFlow()
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)

    score_obs = Score_Observer('ROC-AUC')
    score_obs_pr = Score_Observer('PR-AUC')
    for epoch in range(c.meta_epochs):

        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()
                inputs, labels = preprocess_batch(data)

                z = model(inputs)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                train_loss.append(t2np(loss))
                loss.backward()
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if c.verbose:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        # evaluate
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_z = list()
        test_labels = list()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                z = model(inputs)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                test_z.append(z)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))

        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        score_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                         print_score=c.verbose or epoch == c.meta_epochs - 1)
        # add pr_auc
        precision, recall, thres = precision_recall_curve(is_anomaly, anomaly_score)
        score_obs_pr.update(auc(recall, precision), epoch,
                    print_score=c.verbose or epoch == c.meta_epochs - 1)
        
        # save auc plot
        fpr, tpr, threshold = roc_curve(is_anomaly, anomaly_score)
        plt.clf()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = f'AUC = {roc_auc_score(is_anomaly, anomaly_score):0.2f}')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(f'normflow_auc_plot_{c.class_name}_{epoch}')

        # save fpr & tpr for auc plot
        df_roc_auc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        df_roc_auc.to_csv(f'normflow_fprtpr_{c.class_name}_{epoch}.csv', index=False)

        # save model
        model.to('cpu')
        save_model(model, c.modelname)
        save_weights(model, c.modelname)

    if c.grad_map_viz:
        export_gradient_maps(model, test_loader, optimizer, -1)

    return model
