import numpy as np
import pandas as pd

from river import drift
from skmultiflow.drift_detection import PageHinkley
from scipy import stats

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    
    def psi(expected_array, actual_array, buckets):
        
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            
            return input
        
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
        
        def sub_psi(e_perc, a_perc):
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
 
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value
 
        for i in range(0, len(expected_percents)):
            psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]))
                               
        return psi_value

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        psi_values = psi(expected, actual, buckets)

    return psi_values

def psi_list(train, test):
    """
    Compares the distribution of the target variable in the test dataset to a training data set that was used to develop the model
    
    Parameters
    ----------
    train: pd.DataFrame
    test: pd.DataFrame
    """
    
    psi_list = []
    top_feature_list=train.columns
    
    large = []
    slight = []
    
    for feature in top_feature_list:
        # Assuming you have a validation and training set
        psi_t = calculate_psi(train[feature], test[feature])
        psi_list.append(psi_t)      
        print('Stability index for column',feature,'is',psi_t)
        if(psi_t <= 0.1):
            print('There is no change or shift in the distributions of both datasets for column {}.\n'.format(feature))
        elif(psi_t > 0.2):
            print('This indicates a large shift in the distribution has occurred between both datasets for column {}.\n'.format(feature))
            large.append(feature)
        else:
            print('This indicates a slight change or shift has occurred for column {}.\n'.format(feature))
            slight.append(feature)
            
    if (len(large) == 0) and (len(slight) == 0):
        print("There is no change or shift in the distributions of both datasets for all columns")
        
    if (len(large) != 0):
        print("There is/are indications that a large shift has occurred between both datasets for column {}".format(large))
        
    if (len(slight) != 0):
        print("There is/are indications that a slight shift has occurred between both datasets for column {}".format(slight))
            
    return large, slight

def PageHinkley(train, test):
    """
    Detects data drift by computing the observed values and their mean up to the current moment. Page-Hinkley does not signal warning zones, only change detections.
    This detector implements the CUSUM control chart for detecting changes. This implementation also supports the two-sided Page-Hinkley test to detect increasing and decreasing changes in the mean of the input values.
    
    Parameters:
    ----------
    train: pd.DataFrame()
    test: pd.DataFrame()
    """
    
    # Initialize Page-Hinkley
    ph = drift.PageHinkley()
    
    index_col = []
    col_col = []
    val_col = []
    
    # Update drift detector and verify if change is detected
    for col in train.columns:
        data_stream=[]
        a = np.array(train[col])
        b = np.array(test[col])
        data_stream = np.concatenate((a,b))
        for i, val in enumerate(data_stream):
            in_drift, in_warning = ph.update(val)
            if in_drift:
                print(f"Change detected at index {i} for column: {col} with input value: {val}")
                index_col.append(i)
                col_col.append(col)
                val_col.append(val)
    
    ph_df = pd.DataFrame(data = {
        'index': index_col,
        'column': col_col,
        'value': val_col
    })
    
    if (len(ph_df['column']) != 0):
        print()
        print("In summary, there is/are data drift happenning at columns [column name, frequencies]:")
        print(ph_df['column'].value_counts())
    else:
        print()
        print('There is no data drift detected at all columns')
    
    return ph_df

def ks(train, test, p_value = 0.05):
    """
    The K-S test is a nonparametric test that compares the cumulative distributions of two data sets, in this case, the training data and the post-training data. The null hypothesis for this test states that the data distributions from both the datasets are same. If the null is rejected then we can conclude that there is adrift in the model.
    
    Parameters:
    ----------
    train: pd.DataFrame()
    test: pd.DataFrame()
    p_value: float, defaults to 0.05
    """
    rejected_cols = []
    p_vals = []
    
    for col in train.columns:
        testing = stats.ks_2samp(train[col], test[col])
        p_values = testing[1].round(decimals=4)
        if testing[1] < p_value:
            p_values = testing[1].round(decimals=4)
            print("Result: Column rejected", col, 'at p-value', p_values)
            rejected_cols.append(col)
        p_vals.append(p_values)

    print(f"At {p_value}, we rejected {len(rejected_cols)} column(s) in total")
    
    ks_df = pd.DataFrame(data = {
        'columns': train.columns.tolist(),
        'p_values': p_vals
    })
    
    return ks_df, rejected_cols