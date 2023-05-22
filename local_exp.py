import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers

from qii.qii import QII
from qii.predictor import QIIPredictor
from qii.qoi import QuantityOfInterest

import dalex as dx

from IPython.display import HTML, display
import warnings


def dice_exp(X_train, y_train, model, target, backend = 'sklearn'):
    '''
    Initialize dice experiment CF.
    
    Parameters
    ----------
    X_train: DataFrame,
    y_train: DataFrame or Series, contains target column and list of target variables
    model: model object, can be sklearn, tensorflow, or keras
    target: str, name of target variable
    backend: str, "TF1" ("TF2") for TensorFLow 1.0 (2.0), "PYT" for PyTorch implementations, "sklearn" for Scikit-Learn implementations of standard DiCE (https://arxiv.org/pdf/1905.07697.pdf). For all other frameworks and implementations, provide a dictionary with "model" and "explainer" as keys, and include module and class names as values in the form module_name.class_name. For instance, if there is a model interface class "XGBoostModel" in module "xgboost_model.py" inside the subpackage dice_ml.model_interfaces, and dice interface class "DiceXGBoost" in module "dice_xgboost" inside dice_ml.explainer_interfaces, then backend parameter should be {"model": "xgboost_model.XGBoostModel", "explainer": dice_xgboost.DiceXGBoost}.
    '''
    
    train_dataset = pd.concat([X_train, y_train],axis=1)
    
    cont = X_train.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    cat = X_train.select_dtypes(exclude = ['int64', 'float64']).columns.tolist()
    
    d = dice_ml.Data(dataframe = train_dataset, continuous_features = cont,
                    categorical_features = cat, outcome_name = target)
    
    m = dice_ml.Model(model = model, backend = backend)
    exp = dice_ml.Dice(d, m, method = 'random')
    return exp

def exp_cf(X, exp, total_CFs, desired_range = None, desired_class = 'opposite', features_to_vary = 'all', model_type = 'classification'):
    '''
    Generates counterfactuals
    
    Parameters
    ----------
    X: dataframe, rows of dataset to be analyzed via CF
    exp: object created by dice_exp()
    total_CFs: int, number of CFs to be generated
    desired_range: list of shape (2,), range of desired output for regression case
    desired_class: int or str. Desired CF class - can take 0 or 1. Default value is 'opposite' to the outcome class of query_instance for binary classification. Specify the class name for non-binary classification.
    '''
    
    try:
        if model_type == 'classification':
            e = exp.generate_counterfactuals(X, total_CFs = total_CFs, desired_class = desired_class,
                                            features_to_vary = features_to_vary)
            e.visualize_as_dataframe(show_only_changes = True)
            return e
        elif model_type == 'regression':
            e = exp.generate_counterfactuals(X, total_CFs = total_CFs, desired_range = desired_range,
                                            features_to_vary = features_to_vary)
            e.visualize_as_dataframe(show_only_changes = True)
            return e
        else:
            print('Not compatible model.')
    except Exception:
        print('No counterfactuals found for any of the query points! Kindly check your configuration.')

class Predictor(QIIPredictor):
    '''For QII purpose.'''
    def __init__(self, predictor):
        super(Predictor, self).__init__(predictor)
        
    def predict(self, x):
        # predict the label for instance x
        return self._predictor.predict(x)

def get_feature_names(column_transformer, cat_cols):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans, cat_cols):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['y%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                
                return [name + "__" + f for f in column]
        # print(trans)
        f = [name + "__" + f for f in trans.get_feature_names(cat_cols)]
        # print(f)
        return f 
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
        # print(l_transformers)
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
        #print(l_transformers)
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans, cat_cols)
            #print(_names)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans, cat_cols))
    
    return feature_names

def exp_qii(model, preprocessor, X, idx, cat_cols, method = 'banzhaf'):
    '''
    model: model object, must be model only and no preprocessing steps beforehand
    preprocessor: preprocessor object
    X: DataFrame
    idx: int, row of DataFrame/numpy.ndarray object to be observed
    cat_cols: categorical columns in the X dataset
    '''
    
    predictor = Predictor(model)
    quantity_of_interest = QuantityOfInterest()
    X_proc = preprocessor.transform(X)
    n_features = X_proc.shape[1]
    
    qii = QII(X_proc, n_features, quantity_of_interest)
    
    idx = idx
    x_0 = X_proc[idx:idx + 1]
    
    print(f'QII with {method} method:')
    vals = qii.compute(x_0 = x_0, predictor = predictor, 
                       show_approx = True, evaluated_features = None,
                       data_exhaustive =True, feature_exhaustive = False,
                       method = method)
    print(f'{method}: \n{vals}\n\n')
    
    try:
        feature_names = get_feature_names(preprocessor, cat_cols)
    except:
        p_ind = preprocessor[-1].get_support(indices = True)
        fn = get_feature_names(preprocessor[0], cat_cols)
        feature_names = [fn[x] for x in p_ind]
        
    vals2 = list(vals.values())
    
    vals_df = pd.DataFrame(data = {
        'columns': feature_names,
        'values': vals2
    })
    
    fig, ax = plt.subplots(figsize = (round(len(feature_names) * 0.5), 10))
    ax.bar(feature_names, vals2)
    ax.set_xticklabels(feature_names, rotation = 15, ha = 'right', fontdict = {'fontsize': 8})
    ax.autoscale(enable = True)
    plt.show()
    
    return vals_df

def dalex_exp(model, X_train, y_train, X_test, idx):
    '''
    Explanation object.
    
    model: model object, full model, can be with preprocessing steps
    X_train: DataFrame
    y_train: DataFrame
    X_test: DataFrame
    '''
    
    exp = dx.Explainer(model, X_train, y_train)
    exp.predict(X_test)
    
    obs = X_test.iloc[idx, :]
    
    return exp, obs

def break_down(exp, obs, order = None):
    if order:
        # Additive, only these variables in the order
        bd = exp.predict_parts(obs, type = 'break_down', order = np.array(order))
    else:
        # All variables
        bd = exp.predict_parts(obs, type = 'break_down')
    bd.plot()
    display(bd.result)
    return bd.result

def interactive(exp, obs, count = 10):
    inter = exp.predict_parts(obs, type = 'break_down_interactions', interaction_preference = count)
    inter.plot()
    display(inter.result)
    return inter.result

def profiles(exp, obs, prof_vars, num_cols):
    """prof_vars must be numerical variables only"""
    
    for var in prof_vars:
        if var in num_cols:
            continue
        else:
            print('Numerical variables only!')
    
    cp_profile = exp.predict_profile(obs)
    display(cp_profile.result)
    cp_profile.plot(variables = prof_vars)
    
    return cp_profile.result
    
