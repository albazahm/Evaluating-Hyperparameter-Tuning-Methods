import pytest
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from LGBM_Class import Lgbmclass
from LGBM_Class import train_X, train_y, PARAM_GRID, H_SPACE, EARLY_STOPPING_ROUNDS, N_FOLDS, SEED, NUM_BOOST_ROUNDS, MAX_EVALS, OBJECTIVE_LOSS
from timeit import default_timer as timer
import hyperopt
from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, Trials, fmin
import optuna.integration.lightgbm as lgbo
import optuna
import random

#DEFINING TEST SPACE
TEST_PARAMS = hyperopt.pyll.stochastic.sample(H_SPACE, rng = np.random.RandomState(42))
TEST_PARAMS['subsample'] = TEST_PARAMS['boosting_type']['subsample']
TEST_PARAMS['boosting_type'] = TEST_PARAMS['boosting_type']['boosting_type']
for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_data_in_leaf',
                               'max_bin', 'bagging_freq']:
            TEST_PARAMS[parameter_name] = int(TEST_PARAMS[parameter_name])
OPTIM_TYPE = 'hyperopt'

class TestLGBM():
    """
    Class that compiles unit tests for the class instance as defined in LGBM_Class.py
    """
    def test_crossval(self):
        """
        Tests the cross-validation function in the LGBM_Class
        """
        #OBJECT ORIENTED
        obj_cv = Lgbmclass(train_X, train_y)
        obj_result_list = obj_cv.lgb_crossval(params = TEST_PARAMS, optim_type = OPTIM_TYPE)

        cv_train = lgb.Dataset(data = train_X, label = train_y)

        #REGULAR
        start = timer()
        if OPTIM_TYPE == 'optuna':
            cv_result = lgbo.cv(TEST_PARAMS, obj_cv.train_set, num_boost_round=NUM_BOOST_ROUNDS,
                                 nfold=N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                 metrics=['auc', 'binary', 'xentropy'], seed=SEED)
        else:
            cv_result = lgb.cv(TEST_PARAMS, obj_cv.train_set, num_boost_round=NUM_BOOST_ROUNDS,
                        nfold=N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        metrics=['auc', 'binary', 'xentropy'], seed=SEED)

        run_time = timer() - start

        n_estimators = int(np.argmax(cv_result['auc-mean']) + 1)
        best_score = np.max(cv_result['auc-mean'])
        loss = 1 - best_score
        result_list = (loss, TEST_PARAMS, n_estimators, run_time)
        
        assert obj_result_list[0] == result_list[0], 'loss'
        assert obj_result_list[1] == result_list[1], 'TEST_PARAMS'
        assert obj_result_list[2] == result_list[2], 'n_estimators'
        assert type(obj_result_list[3]) == type(result_list[3]), 'time'

    def test_hyperopt(self):
        """
        Tests the Hyperopt optimizaton method in LGBM_Class
        """
        #OBJECTIVE ORIENTED
        train_set = lgb.Dataset(data = train_X, label = train_y)
        obj_lgb = Lgbmclass(train_X, train_y)
        obj_result_list = obj_lgb.hyperopt_space()


        #REGULAR
        space, algo, trials = H_SPACE, tpe.suggest, Trials()

        def hyperopt_obj(space):

            """
            Defines some of the Hyperopt parameter space and objective function
            """
            subsample = space['boosting_type'].get('subsample', 1.0)
            space['boosting_type'] = space['boosting_type']['boosting_type']
            space['subsample'] = subsample
            for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_data_in_leaf',
                               'max_bin', 'bagging_freq']:
                space[parameter_name] = int(space[parameter_name])
            cv_result = lgb.cv(space, train_set, num_boost_round=NUM_BOOST_ROUNDS,
                        nfold=N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        metrics=['auc', 'binary', 'xentropy'], seed=SEED)
            best_score = np.max(cv_result['auc-mean'])
            loss = 1 - best_score
            return {'loss' : loss, 'status': STATUS_OK}
        
        result = fmin(fn=hyperopt_obj, space=space, algo=algo, max_evals=MAX_EVALS,
                          trials=trials, rstate=np.random.RandomState(SEED))

        result_list = (result, trials)
        assert len(obj_result_list) == len(result_list)
        assert obj_result_list[0] == result_list[0]

    def test_optuna(self):
        """
        Tests the Optuna optimizaton method in LGBM_Class
        """
        #OBJECT ORIENTED
        train_set = lgb.Dataset(data = train_X, label = train_y)
        obj_lgb = Lgbmclass(train_X, train_y)
        obj_result = obj_lgb.optuna_space()

        #REGULAR
        def optuna_obj(trial):
        """
        Defines the optuna parameter space and objective function
        """
            space = {
            'num_leaves': trial.suggest_int('num_leaves', 16, 196, 4),
            'max_bin' : trial.suggest_int('max_bin', 63, 255, 64),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 20, 500),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            # removed 'dart'
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.05, 0.25),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin',20000, 300000, 20000),
            'feature_fraction': trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 7),
            'verbosity' : 0,
            'objective' : OBJECTIVE_LOSS
                }

            for parameter_name in ['num_leaves', 'min_data_in_leaf',
                               'max_bin', 'bagging_freq']:
                space[parameter_name] = int(space[parameter_name])

            if space['boosting_type'] == 'goss':
                space['subsample'] = 1
            else:
                space['subsample'] = trial.suggest_uniform('subsample', 0.5, 1)

            cv_result = lgb.cv(space, train_set, num_boost_round=NUM_BOOST_ROUNDS,
                        nfold=N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        metrics=['auc', 'binary', 'xentropy'], seed=SEED)
            best_score = np.max(cv_result['auc-mean'])
            loss = 1 - best_score
            return loss
        
        study = optuna.create_study(direction='minimize', sampler = optuna.samplers.TPESampler(seed = SEED))
        study.optimize(optuna_obj, n_trials=MAX_EVALS)

        assert obj_result.best_params == study.best_params

    def test_random_search(self):
        """
        Tests the Random Search optimizaton method in LGBM_Class
        """
        random.seed(SEED)
        #OBJECT ORIENTED
        train_set = lgb.Dataset(data = train_X, label = train_y)
        obj_lgb = Lgbmclass(train_X, train_y)
        obj_result = obj_lgb.random_space()
        
        #REGULAR
        space = PARAM_GRID
        random_results = pd.DataFrame(columns=['loss', 'params', 'estimators'],
                                     index=list(range(MAX_EVALS)))

        def rand_obj(space):
            """
            Defines some of the random search parameter space and objective function
            """
            
            subsample_dist = list(np.linspace(0.5, 1, 100))

            if space['boosting_type'] == 'goss':
                space['subsample'] = 1.0
            else:
                space['subsample'] = random.sample(subsample_dist, 1)[0]

            cv_result = lgb.cv(space, train_set, num_boost_round=NUM_BOOST_ROUNDS,
                    nfold=N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    metrics=['auc', 'binary', 'xentropy'], seed=SEED)
            
            best_score = np.max(cv_result['auc-mean'])
            loss = 1 - best_score
            n_estimators = int(np.argmax(cv_result['auc-mean']) + 1)

            return [loss, params, n_estimators]

        for i in range(MAX_EVALS):

            params = {key: random.sample(value, 1)[0] for key, value in space.items()}
            results_list = rand_obj(params)
            random_results.loc[i, :] = results_list
        
        random_results.sort_values('loss', ascending = True, inplace = True)

        assert (obj_result.loc[:, ['loss', 'params', 'estimators']].reset_index(drop = True) == random_results.reset_index(drop = True)).sum().any()

    def test_plots(self):
        """
        Tests successfull creating of plots in the LGBM_Class
        """
        obj_lgb = Lgbmclass(train_X, train_y)
        obj_lgb.hyperopt_space()
        obj_lgb.train(train_X, train_y)
        obj_lgb.evaluate()

        assert os.path.exists('./feature_importance.png')
        assert os.path.exists('./roc.png')
        assert os.path.exists('./prcurve.png')
        assert os.path.exists('./fpr-fnr.png')
        

        



        

