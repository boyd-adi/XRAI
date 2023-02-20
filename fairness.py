

import sklearn
import pickle
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# XRAI features
from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights
from responsibleai.feature_metadata import FeatureMetadata
from raiwidgets.cohort import Cohort, CohortFilter, CohortFilterMethods
import seaborn as sns 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import joblib
import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import json
from IPython.display import display, HTML
import warnings
warnings.filterwarnings("ignore")

# Standard Data Science Helpers
import numpy as np
import pandas as pd
import scipy

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import iplot, init_notebook_mode
#init_notebook_mode(connected=True)

# import cufflinks as cf
# cf.go_offline(connected=True)
# cf.set_config_file(colorscale='plotly', world_readable=True)

# Extra options
pd.options.display.max_rows = 30
pd.options.display.max_columns = 25

# Show all code cells output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

def model_performance(model,test_x,test_y,train_x,train_y,all_test,all_train, target_feature ,previlaged_groups):
    
   
    
    # Define a empty DataFrame
    #DF=pd.DataFrame()
    DF_test=pd.DataFrame()
    DF_train=pd.DataFrame()
    
    for pg in previlaged_groups.keys():
        #
        DF_test[pg]=test_x[pg]
        DF_test["Ground_truth"]=pd.DataFrame(test_y)
        DF_test["Predicted"]=model.predict(test_x)


        #
        DF_train[pg]=train_x[pg]
        DF_train["Ground_truth"]=pd.DataFrame(train_y)
        DF_train["Predicted"]=model.predict(train_x)
    
    # Metrics
    

    ### Test
    print("Performance on test data :\n")
    print("Accuracy on test data: ", round(accuracy_score(DF_test["Ground_truth"], DF_test["Predicted"]),3))
    p=sns.pairplot(all_test, hue =target_feature )
    
    conmat = confusion_matrix(DF_test["Ground_truth"], DF_test["Predicted"])
    val = np.mat(conmat) 
    classnames = list(set(DF_test["Ground_truth"]))
    df_cm = pd.DataFrame(val, index=classnames, columns=classnames,)
    
    
    plt.figure()
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.1f', cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(' Model Results')
    plt.show()    
    
    
    ### Train 
    print("Performance on Train data :\n")
    print("Accuracy on test data: ", round(accuracy_score(DF_train["Ground_truth"], DF_train["Predicted"]),3))
    p=sns.pairplot(all_train, hue =target_feature )
    conmat = confusion_matrix(DF_train["Ground_truth"], DF_train["Predicted"])
    val = np.mat(conmat) 
    classnames = list(set(DF_train["Ground_truth"]))
    df_cm = pd.DataFrame(val, index=classnames, columns=classnames,)
    
    
    plt.figure()
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.0g',cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Model Results')
    plt.show()    
    
    
    
    for pg in previlaged_groups.keys():
    
        ## This function is designed to calculate metrics on different splits of defined previlaged groupes

        def Result_sum(DF_test,pg):

            datatest={'Split': ['Overall performance'],
            "previlaged_groups":[pg],
            'Accuracy': [metrics.accuracy_score(DF_test["Ground_truth"],DF_test["Predicted"])],
             "F1 Score":[metrics.f1_score(DF_test["Ground_truth"],DF_test["Predicted"])],
             "Precision Score":[metrics.precision_score(DF_test["Ground_truth"],DF_test["Predicted"])],
             "Recall Score": [metrics.recall_score(DF_test["Ground_truth"],DF_test["Predicted"])] }
            result_sum_test=pd.DataFrame(datatest)


            for i in np.unique(DF_test[pg]):

                DFI_test=DF_test.groupby(pg).get_group(i)
                result_sum_test.loc[len(result_sum_test)]=[str(i),pg,metrics.accuracy_score(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.f1_score(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.precision_score(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.recall_score(DFI_test["Ground_truth"],DFI_test["Predicted"])]


            return result_sum_test



        result_sum_test=Result_sum(DF_test,pg)
        result_sum_train=Result_sum(DF_train,pg)
        limit_on_plot_number=5

        number_of_groups=len(np.unique(result_sum_test["Split"]))
        if number_of_groups>5:
            limit_on_plot_number=5
        else:
            limit_on_plot_number=number_of_groups
            
    

        # Plot metrics for different groups
        plt.bar(result_sum_test["Split"][:limit_on_plot_number], result_sum_test["Accuracy"][:limit_on_plot_number], color ='blue',width = 0.4,label="Test")
        plt.bar(result_sum_train["Split"][:limit_on_plot_number], result_sum_train["Accuracy"][:limit_on_plot_number], color ='red',width = 0.4, label="Train")
        plt.xlabel(pg)
        plt.ylabel("Accuracy")
        plt.xticks(rotation = 90)
        plt.legend()
       
        plt.show()

    return result_sum_train

    

def fairness(model,x,y,previlaged_groups={}):
    
    
    x=x.copy()
    y=y.copy()
    
    x["y_prime"]=list(model.predict(x))
    x["y"]=list(y)
    
    
  # function for calculating fainess metrics
    def fairness_metrics(y,y_prime):
        """Calculate fairness for subgroup of population"""
    
        #Confusion Matrix
        cm=confusion_matrix(list(y),list(y_prime))
        
        
        TN, FP, FN, TP = cm.ravel()[0], cm.ravel()[1], cm.ravel()[2], cm.ravel()[3]
    
        N = TP+FP+FN+TN #Total population
        ACC = (TP+TN)/N #Accuracy
        TPR = TP/(TP+FN) # True positive rate
        FPR = FP/(FP+TN) # False positive rate
        FNR = FN/(TP+FN) # False negative rate
        PPP = (TP + FP)/N # % predicted as positive
    
        return np.array([ACC, TPR, FPR, FNR, PPP])
    
    
    

    print('''In the US there is a legal precedent to set the cutoff to 0.8. That is the predicted as positive for the unprivileged group must not be less than 80% of that of the privileged group.''')
    
    print('------------------------------------------------------------------------------------------------------------------------')
    
    
    # Calculating overall Disparity  
    i=1
    fairness_report=pd.DataFrame(columns=["Variable","Previledge_group","Accuracy","True_positive_rate","False_positive_rate","False_negative_rate","predicted_as_positive"])    
    #Calculate fairness metrics 
    fm = fairness_metrics(x['y'],x["y_prime"])
    a=["Overall",""]
    b=list(fm)
    b=[round(item, 3) for item in b]
    
    a.extend(b)
    row=a
    fairness_report.loc[0,:]=row
    
    for i in previlaged_groups.keys():
        
        
        # Find and mark protected and unprotected
        x["Prev_"+i]=[1 if r==previlaged_groups[i] else 0 for r in x[i]]
        
        #  make a list of protected features and fairness metrics for each
        a=[i,previlaged_groups[i]]
        b=list(fairness_metrics(x.loc[x["Prev_"+i]==1,'y'],x.loc[x["Prev_"+i]==1,"y_prime"]))
        b=[round(item, 3) for item in b]
       
        a.extend(b)
        row=a

        
        # Add metric lists to the fairness_report data frame to use it later in model artifacts
        fairness_report.loc[i,:]=row
        
        #Calculate fairness metrics for previledge group
        fm_Prev_1 = fairness_metrics(x.loc[x["Prev_"+i]==1,'y'],x.loc[x["Prev_"+i]==1,"y_prime"])[-1]
        fm_Prev_0 = fairness_metrics(x.loc[x["Prev_"+i]==0,'y'],x.loc[x["Prev_"+i]==0,"y_prime"])[-1]
        
        #Get ratio of fairness metrics
        fm_ratio = fm_Prev_0/fm_Prev_1
        
        
        print("----------------------------------------------------{}---------------------------------------------------------".format(i))
        print("For the previlage feature {} of {} the disparity index is {}".format(i,previlaged_groups[i],round(fm_ratio,3)))
        print("\n")
        
        if fm_ratio>0.8:
            
            print('\x1b[6;30;42m'+"The model is fair towards the {} in {}".format(previlaged_groups[i],i)+ '\x1b[0m')
            
        else:
            
            print('\x1b[6;30;41m'+"The model is NOT fair towards the {} in {}".format(previlaged_groups[i],i)+ '\x1b[0m')
    
    display(fairness_report)
    #sns.barplot(data=fairness_report.T)

    
