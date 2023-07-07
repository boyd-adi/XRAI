# Explainer Dashboard

A custom dashboard was developed to showcase the capabilities of the functions of the XRAI tTlkit in an interactive and easy to understand environment. 

## How to launch notebook and run the dashboard 
1.	Ensure that python libraries are installed in your environment accordance with `requirements.txt`.
2.	Start the “Dashboard.ipynb” file
3.	Enter your training and test datasets and model in the notebook
4.	Run the explainerdashboard function to start the dashboard 

For any troubles in launching the dashboard, please check the following guidelines:
1. Dataset and Model Ingestion: Ensure that the model (.sav or .pkl) and dataset (.csv) is properly loaded. The `load_data_model` function from the `data_model` module provides a convenient way to do this.
2. Create an `explainerdashboard` explainer instance using the loaded model and test data. If this shows an error, this means that the model or data was not compatible with the `explainerdashboard` library. However, as the explainer instance won't really be used to extract data from, a dummy explainer can be built using sklearn's `LinearRegression` and an empty pandas' dataframes.
3. Extract the preprocessing pipeline in your model. Improper extraction of the preprocessing pipeline may lead to QII and SHAP components not working properly. Set to `None` if no preprocessing steps was present in the model pipeline.
4. Configure the groupings for the Grouped Variable Importance, if you want to analyze the variable importances on your dataset divided in a specific groupings.
5. Separate the continuous and categorical variables.
6. The model should be in a dictionary form, so set a name for the model as key, and the model itself as the value.
7. Select the `model_type` properly, either regression or classification. Incorrect assignment of this variable may lead to wrong components being displayed, which will not work since the `model_type` is not consitent with the model.
8. If the pipeline used was made thru sklearn, set `is_sklearn_pipe` to True, otherwise False.
9. Create a Dalex explainer object, which will be used by the Dalex-related components. This may take a while for larger datasets, so running this once in the notebook will save time, instead of running it every time in the dashboard for each Dalex-related components.
10. After the assignments of the necessary inputs, the dashboard can now be run by feeding the necessay inputs per tab. Click the `http://192.168.100.24:8050` to view the custom dashboard in another tab.

## Assumptions and Limitations
1.	For classification and regression models, users may need to create objects for pipelines
2.	Train and test data should be in csv format
3.	Train data and Test data should be in same format. For example, if you have applied label encoding on your categorical variable you should have done the same processing on your same variable in test data.
4.	For large datasets, some functions may take upwards of 30 minutes to complete.

## For further understanding
1.	The XRAI Toolkit is complemented by the XRAI Guidelines document. Please refer to the this document for a theoretical understanding of the XRAI principles.  
2.	The various functions are powered by the different .py files. Please read the documentation on them for further information. 