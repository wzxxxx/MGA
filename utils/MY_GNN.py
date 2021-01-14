import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import pandas as pd
from utils import weight_visualization


class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight=return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self, bg, feats):
        feat_list = []
        atom_list = []
        # cal specific feats
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # cal shared feats
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')
        # feat_list.append(shared_feats_sum)
        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            else:
                return feat_list
        else:
            return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )


class MLPBinaryClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPBinaryClassifier, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, h):
        return self.predict(h)


class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels=64*21, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                               num_bases=None, bias=True, activation=activation,
                                               self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, norm=None):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        torch.cuda.empty_cache()
        return new_feats


class BaseGNN(nn.Module):
    def __init__(self, gnn_out_feats, n_tasks, rgcn_drop_out=0.5, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(BaseGNN, self).__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)
        self.fc_in_feats = gnn_out_feats
        self.return_mol_embedding=return_mol_embedding

        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self, bg, node_feats, etype, norm=None):
        # Update atom features with GNNs
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype, norm)

        # Compute molecule features from atom features
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        for i in range(self.task_num):
            # mol_feats = torch.cat([feats_list[-1], feats_list[i]], dim=1)
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return feats_list[0]
        else:
            # generate atom weight and atom feats
            if self.return_weight:
                return prediction_all, atom_weight_list, node_feats
            # just generate prediction
            return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class MGA(BaseGNN):
    def __init__(self, in_feats, rgcn_hidden_feats, n_tasks, return_weight=False,
                 classifier_hidden_feats=128, loop=False, return_mol_embedding=False,
                 rgcn_drop_out=0.5, dropout=0.):
        super(MGA, self).__init__(gnn_out_feats=rgcn_hidden_feats[-1],
                                  n_tasks=n_tasks,
                                  classifier_hidden_feats=classifier_hidden_feats,
                                  return_mol_embedding=return_mol_embedding,
                                  return_weight=return_weight,
                                  rgcn_drop_out=rgcn_drop_out,
                                  dropout=dropout,
                                  )

        for i in range(len(rgcn_hidden_feats)):
            out_feats = rgcn_hidden_feats[i]
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pos_weight(train_set, classification_num):
    smiles, graphs, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    for task in range(classification_num):
        num_pos = 0
        num_impos = 0
        for i in labels[:, task]:
            if i == 1:
                num_pos = num_pos + 1
            if i == 0:
                num_impos = num_impos + 1
        weight = num_impos / (num_pos+0.00000001)
        task_pos_weight_list.append(weight)
    task_pos_weight = torch.tensor(task_pos_weight_list)
    return task_pos_weight


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()


def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)

    return smiles, bg, labels,  mask


def run_a_train_epoch_heterogeneous(args, epoch, model, data_loader, loss_criterion_c, loss_criterion_r, optimizer, task_weight=None):
    model.train()
    train_meter_c = Meter()
    train_meter_r = Meter()
    if task_weight is not None:
        task_weight = task_weight.float().to(args['device'])

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        mask = mask.float().to(args['device'])
        labels.float().to(args['device'])
        atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
        bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
        logits = model(bg, atom_feats, bond_feats, norm=None)
        labels = labels.type_as(logits).to(args['device'])
        # calculate loss according to different task class
        if args['task_class'] == 'classification_regression':
            # split classification and regression
            logits_c = logits[:,:args['classification_num']]
            labels_c = labels[:,:args['classification_num']]
            mask_c = mask[:,:args['classification_num']]

            logits_r = logits[:,args['classification_num']:]
            labels_r = labels[:,args['classification_num']:]
            mask_r = mask[:,args['classification_num']:]
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float()).mean() \
                       + (loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float()).mean()
            else:
                task_weight_c = task_weight[:args['classification_num']]
                task_weight_r = task_weight[args['classification_num']:]
                loss = (torch.mean(loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float(), dim=0)*task_weight_c).mean() \
                       + (torch.mean(loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float(), dim=0)*task_weight_r).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_c.update(logits_c, labels_c, mask_c)
            train_meter_r.update(logits_r, labels_r, mask_r)
            del bg, mask, labels, atom_feats, bond_feats, loss, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
            torch.cuda.empty_cache()
        elif args['task_class'] == 'classification':
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_c(logits, labels) * (mask != 0).float(),dim=0)*task_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_c.update(logits, labels, mask)
            del bg, mask, labels, atom_feats, bond_feats, loss,  logits
            torch.cuda.empty_cache()
        else:
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_r(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_r(logits, labels) * (mask != 0).float(), dim=0)*task_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_r.update(logits, labels, mask)
            del bg, mask, labels, atom_feats, bond_feats, loss,  logits
            torch.cuda.empty_cache()
    if args['task_class'] == 'classification_regression':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']) +
                              train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], 'r2+auc', train_score))
    elif args['task_class'] == 'classification':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['classification_metric_name'], train_score))
    else:
        train_score = np.mean(train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['regression_metric_name'], train_score))


def run_an_eval_epoch_heterogeneous(args, model, data_loader):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]
                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]
                # Mask non-existing labels
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
                torch.cuda.empty_cache()
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
            else:
                # Mask non-existing labels
                eval_meter_r.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            return eval_meter_c.compute_metric(args['classification_metric_name'])
        else:
            return eval_meter_r.compute_metric(args['regression_metric_name'])


def run_an_eval_epoch_pih(args, model, data_loader, output_path):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    smiles_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            smiles_list = smiles_list + smiles
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]
                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]
                # Mask non-existing labels
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
                torch.cuda.empty_cache()
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
            else:
                # Mask non-existing labels
                eval_meter_r.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            y_pred, y_true = eval_meter_c.compute_metric('return_pred_true')
            result = pd.DataFrame(columns=['smiles', 'pred', 'true'])
            result['smiles'] = smiles_list
            result['pred'] = np.squeeze(y_pred.numpy()).tolist()
            result['true'] = np.squeeze(y_true.numpy()).tolist()
            result.to_csv(output_path, index=None)
        else:
            return eval_meter_r.compute_metric(args['regression_metric_name'])


def run_an_eval_epoch_heterogeneous_return_weight(args, model, data_loader, vis_list=None, vis_task='CYP2D6'):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            #####
            labels = labels.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits, atom_weight_list, node_feats = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])
            logits_c = logits[:, :args['classification_num']]
            logits_c = torch.sigmoid(logits_c)
            # different tasks with different atom weight

            for mol_index in range(len(smiles)):
                atom_smiles = smiles[mol_index]
                if atom_smiles in vis_list:
                    for tasks_index in range(31):
                        # if args['all_task_list'][tasks_index] == vis_task:
                        if labels[mol_index, tasks_index]!=123456:
                            bg.ndata['w'] = atom_weight_list[tasks_index]
                            bg.ndata['feats'] = node_feats
                            unbatch_bg = dgl.unbatch(bg)
                            one_atom_weight = unbatch_bg[mol_index].ndata['w']
                            one_atom_feats = unbatch_bg[mol_index].ndata['feats']
                            # visual selected molecules
                            print('Tasks:', tasks_index, args['all_task_list'][tasks_index], "**********************")
                            if tasks_index < 26:
                                print('Predict values:', logits_c[mol_index, tasks_index])
                            else:
                                print('Predict values:', logits[mol_index, tasks_index])
                            print('True values:', labels[mol_index, tasks_index])
                            weight_visualization.weight_visulize(atom_smiles, one_atom_weight)
                else:
                    continue


def run_an_eval_epoch_heterogeneous_return_weight_py(args, model, data_loader, vis_list=None, vis_task='CYP2D6'):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            #####
            labels = labels.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits, atom_weight_list, node_feats = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])
            logits_c = logits[:, :args['classification_num']]
            logits_c = torch.sigmoid(logits_c)
            # different tasks with different atom weight

            for mol_index in range(len(smiles)):
                atom_smiles = smiles[mol_index]
                if atom_smiles in vis_list:
                    for tasks_index in range(31):
                        if args['all_task_list'][tasks_index] == vis_task:
                            if labels[mol_index, tasks_index]!=123456:
                                bg.ndata['w'] = atom_weight_list[tasks_index]
                                bg.ndata['feats'] = node_feats
                                unbatch_bg = dgl.unbatch(bg)
                                one_atom_weight = unbatch_bg[mol_index].ndata['w']
                                one_atom_feats = unbatch_bg[mol_index].ndata['feats']
                                # visual selected molecules
                                print('Tasks:', tasks_index, args['all_task_list'][tasks_index], "**********************")
                                if tasks_index < 26:
                                    print('Predict values:', logits_c[mol_index, tasks_index])
                                else:
                                    print('Predict values:', logits[mol_index, tasks_index])
                                print('True values:', labels[mol_index, tasks_index])
                                weight_visualization.weight_visulize_py(atom_smiles, one_atom_weight)
                else:
                    continue


def run_an_eval_epoch_heterogeneous_generate_weight(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id+1, len(data_loader)))
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats, bond_feats, norm=None)
            for atom_weight in atom_weight_list:
                atom_list_all.append(atom_weight[args['select_task_index']])
    task_name = args['select_task_list'][0]
    atom_weight_list = pd.DataFrame(atom_list_all, columns=['atom_weight'])
    atom_weight_list.to_csv(task_name+"_atom_weight.csv", index=None)


def generate_chemical_environment(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id + 1, len(data_loader)))
            smiles, bg, labels, mask = batch_data
            print(bg.ndata[args['atom_data_field']][1])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats, bond_feats, norm=None)
            print('after training:', bg.ndata['h'][1])


def generate_mol_feats(args, model, data_loader, dataset_output_path):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            feats = model(bg, atom_feats, bond_feats, norm=None).numpy().tolist()
            feats_name = ['graph-feature' + str(i+1) for i in range(64)]
            data = pd.DataFrame(feats, columns=feats_name)
            data['smiles'] = smiles
            data['labels'] = labels.squeeze().numpy().tolist()
    data.to_csv(dataset_output_path, index=None)


class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            task_name = task_name
            filename ='../model/{}_early_stop.pth'.format(task_name)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked']
        if torch.cuda.is_available():
            pretrained_model = torch.load('../model/'+self.pretrained_model)
        else:
            pretrained_model = torch.load('../model/'+self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    def load_model_attention(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.bias',
                                 'weighted_sum_readout.shared_weighting.0.weight',
                                 'weighted_sum_readout.shared_weighting.0.bias',
                                 ]
        if torch.cuda.is_available():
            pretrained_model = torch.load('../model/' + self.pretrained_model)
        else:
            pretrained_model = torch.load('../model/' + self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)


