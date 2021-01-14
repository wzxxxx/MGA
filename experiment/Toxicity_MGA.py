import numpy as np
from utils import build_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.MY_GNN import collate_molgraphs, EarlyStopping, run_a_train_epoch_heterogeneous, \
    run_an_eval_epoch_heterogeneous, set_random_seed, MGA, pos_weight
import os
import time
import pandas as pd
start = time.time()


# fix parameters of model
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['atom_data_field'] = 'atom'
args['bond_data_field'] = 'etype'
args['classification_metric_name'] = 'roc_auc'
args['regression_metric_name'] = 'r2'
# model parameter
args['num_epochs'] = 500
args['patience'] = 50
args['batch_size'] = 128
args['mode'] = 'higher'
args['in_feats'] = 40
args['rgcn_hidden_feats'] = [64, 64]
args['classifier_hidden_feats'] = 64
args['rgcn_drop_out'] = 0.2
args['drop_out'] = 0.2
args['lr'] = 3
args['weight_decay'] = 5
args['loop'] = True

# task name (model name)
args['task_name'] = 'toxicity'  # change
args['data_name'] = 'toxicity'  # change
args['times'] = 10

# selected task, generate select task index, task class, and classification_num
args['select_task_list'] = ['Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity',
                            'Eye irritation', 'Eye corrosion', 'Cardiotoxicity1', 'Cardiotoxicity5',
                            'Cardiotoxicity10', 'Cardiotoxicity30',
                            'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4',
                            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
                            'Acute oral toxicity (LD50)', 'LC50DM', 'BCF', 'LC50', 'IGC50']  # change
args['select_task_index'] = []
args['classification_num'] = 0
args['regression_num'] = 0
args['all_task_list'] = ['Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity',
                         'Eye irritation', 'Eye corrosion', 'Cardiotoxicity1', 'Cardiotoxicity5',
                         'Cardiotoxicity10', 'Cardiotoxicity30',
                         'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4',
                         'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
                         'Acute oral toxicity (LD50)', 'LC50DM', 'BCF', 'LC50', 'IGC50']  # change
# generate select task index
for index, task in enumerate(args['all_task_list']):
    if task in args['select_task_list']:
        args['select_task_index'].append(index)
# generate classification_num
for task in args['select_task_list']:
    if task in ['Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity',
                'Eye irritation', 'Eye corrosion', 'Cardiotoxicity1', 'Cardiotoxicity5',
                'Cardiotoxicity10', 'Cardiotoxicity30',
                'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4',
                'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'Hepatotoxicity']:
        args['classification_num'] = args['classification_num'] + 1
    if task in ['Acute oral toxicity (LD50)', 'LC50DM', 'BCF', 'LC50', 'IGC50', 'Urinary_tract_toxicity']:
        args['regression_num'] = args['regression_num'] + 1

# generate classification_num
if args['classification_num'] != 0 and args['regression_num'] != 0:
    args['task_class'] = 'classification_regression'
if args['classification_num'] != 0 and args['regression_num'] == 0:
    args['task_class'] = 'classification'
if args['classification_num'] == 0 and args['regression_num'] != 0:
    args['task_class'] = 'regression'
print('Classification task:{}, Regression Task:{}'.format(args['classification_num'], args['regression_num']))
args['bin_path'] = '../data/' + args['data_name'] + '.bin'
args['group_path'] = '../data/' + args['data_name'] + '_group.csv'


result_pd = pd.DataFrame(columns=args['select_task_list']+['group'] + args['select_task_list']+['group']
                         + args['select_task_list']+['group'])
all_times_train_result = []
all_times_val_result = []
all_times_test_result = []
for time_id in range(args['times']):
    set_random_seed(2020+time_id)
    one_time_train_result = []
    one_time_val_result = []
    one_time_test_result = []
    print('***************************************************************************************************')
    print('{}, {}/{} time'.format(args['task_name'], time_id+1, args['times']))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_dataset.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        select_task_index=args['select_task_index']
    )
    print("Molecule graph generation is complete !")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_molgraphs)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    pos_weight_np = pos_weight(train_set, classification_num=args['classification_num'])
    loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
    loss_criterion_r = torch.nn.MSELoss(reduction='none')
    model = MGA(in_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                n_tasks=task_number, rgcn_drop_out=args['rgcn_drop_out'],
                classifier_hidden_feats=args['classifier_hidden_feats'], dropout=args['drop_out'],
                loop=args['loop'])
    optimizer = Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch_heterogeneous(args, epoch, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer)

        # Validation and early stop
        validation_result = run_an_eval_epoch_heterogeneous(args, model, val_loader)
        val_score = np.mean(validation_result)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
            epoch + 1, args['num_epochs'],
            val_score,  stopper.best_score)+' validation result:', validation_result)
        if early_stop:
            break
    stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch_heterogeneous(args, model, test_loader)
    train_score = run_an_eval_epoch_heterogeneous(args, model, train_loader)
    val_score = run_an_eval_epoch_heterogeneous(args, model, val_loader)
    # deal result
    result = train_score + ['training'] + val_score + ['valid'] + test_score + ['test']
    result_pd.loc[time_id] = result
    print('********************************{}, {}_times_result*******************************'.format(args['task_name'], time_id+1))
    print("training_result:", train_score)
    print("val_result:", val_score)
    print("test_result:", test_score)

result_pd.to_csv('../result/'+args['task_name']+'_result.csv', index=None)
elapsed = (time.time() - start)
m, s = divmod(elapsed, 60)
h, m = divmod(m, 60)
print("Time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))












