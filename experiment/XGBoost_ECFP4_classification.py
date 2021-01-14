import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from sklearn import metrics

parameters={}
# space of hyperopt parameters
space = {'max_depth': hp.choice('max_depth', list(range(3,10,1))),
         'min_child_weight': hp.choice('min_child_weight', list(range(1,6,1))),
         'gamma': hp.choice('gamma', [i/50.0 for i in range(10)]),
         'reg_lambda':hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1]),
         'reg_alpha':hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1]),
         'lr':hp.choice('lr', [0.01, 0.05, 0.001, 0.005]),
         'n_estimators':hp.choice('n_estimators', list(range(100, 300, 20))),
         'colsample_bytree':hp.choice('colsample_bytree',[i/100.0 for i in range(75,90,5)]),
         'subsample': hp.choice('subsample', [i/100.0 for i in range(75,90,5)]),
         }

task_list = ['Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity',
             'Eye irritation', 'Eye corrosion', 'Cardiotoxicity1', 'Cardiotoxicity5',
             'Cardiotoxicity10', 'Cardiotoxicity30',
             'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4',
             'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
result_xgb = pd.DataFrame(columns=task_list)
for xgb_graph_feats_task in task_list:
    print('***************************************************************************************************')
    print(xgb_graph_feats_task)
    print('***************************************************************************************************')
    args = {}
    training_set = pd.read_csv(xgb_graph_feats_task+'_ECFP4_training.csv', index_col=None)
    valid_set = pd.read_csv(xgb_graph_feats_task+'_ECFP4_valid.csv', index_col=None)
    test_set = pd.read_csv(xgb_graph_feats_task+'_ECFP4_test.csv', index_col=None)
    x_colunms = [x for x in training_set.columns if x not in ['smiles', 'labels']]
    label_columns = ['labels']
    train_x = training_set[x_colunms]
    train_y = training_set[label_columns].values.ravel()
    valid_x = valid_set[x_colunms]
    valid_y = valid_set[label_columns].values.ravel()
    test_x = test_set[x_colunms]
    test_y = test_set[label_columns].values.ravel()


    def hyperopt_my_xgb(parameter):
        model = xgb.XGBClassifier(learning_rate=parameter['lr'], max_depth=parameter['max_depth'],
                                  min_child_weight=parameter['min_child_weight'], gamma=parameter['gamma'],
                                  reg_alpha=parameter['reg_alpha'], reg_lambda=parameter['reg_lambda'],
                                  subsample=parameter['subsample'], colsample_bytree=parameter['colsample_bytree'],
                                  n_estimators=parameter['n_estimators'], random_state=2020, n_jobs=6)
        model.fit(train_x, train_y, eval_metric='auc')

        # valid set
        valid_prediction = model.predict_proba(valid_x)[:, 1]
        auc = metrics.roc_auc_score(valid_y, valid_prediction)
        return {'loss':-auc, 'status':STATUS_OK, 'model':model}


    # hyper parameter optimization
    trials = Trials()
    best = fmin(hyperopt_my_xgb, space, algo=tpe.suggest, trials=trials, max_evals=50)
    print(best)

    # load the best model parameters
    args['max_depth'] = list(range(3,10,1))[best['max_depth']]
    args['min_child_weight'] = list(range(1,6,1))[best['min_child_weight']]
    args['gamma'] = [i/50 for i in range(10)][best['gamma']]
    args['reg_lambda'] = [1e-5, 1e-2, 0.1, 1][best['reg_lambda']]
    args['reg_alpha'] = [1e-5, 1e-2, 0.1, 1][best['reg_alpha']]
    args['lr'] = [0.01, 0.05, 0.001, 0.005][best['lr']]
    args['n_estimators'] = list(range(100, 300, 20))[best['n_estimators']]
    args['colsample_bytree'] = [i / 100.0 for i in range(75, 90, 5)][best['colsample_bytree']]
    args['subsample'] = [i / 100.0 for i in range(75, 90, 5)][best['subsample']]

    result = []
    for i in range(10):
        model = xgb.XGBClassifier(learning_rate=args['lr'], max_depth=args['max_depth'],
                                  min_child_weight=args['min_child_weight'], gamma=args['gamma'],
                                  reg_alpha=args['reg_alpha'], reg_lambda=args['reg_lambda'],
                                  subsample=args['subsample'], colsample_bytree=args['colsample_bytree'],
                                  n_estimators=args['n_estimators'], seed=2020+i, n_jobs=6)
        model.fit(train_x, train_y.ravel())
        test_prediction = model.predict_proba(test_x)[:, 1]
        auc = metrics.roc_auc_score(test_y, test_prediction)
        result.append(auc)
    result_xgb[xgb_graph_feats_task] = result
    result_pd = pd.DataFrame(result)
    result_pd.to_csv(xgb_graph_feats_task+'_ECFP4_xgb_result.csv', index=None)
    parameters[str(xgb_graph_feats_task)]=args
result_xgb.to_csv('XGB_ECFP4_result.csv', index=None)
filename = open('xgb_ECFP4_parameters.txt', 'w')
for k,v in parameters.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()


