# Explainable and Responsible AI Toolkit (v2.0 - 2023-05)
***Developed by AI COE and DS Consulting***

The **DSAI Explainable and Responsible AI (XRAI) Toolkit** is a complementary tool to the [XRAI Guidelines document](https://unionbankphilippines.sharepoint.com/:b:/s/DataScienceInsightsTeam/EbGWZEJkn7REt1zzHspu-xABsLDpD1eD6mgHMjPJypnzdA?e=wm55U7). The Toolkit provides a one-stop tool for technical tests were identified by packaging widely used open-source libraries into a single platform, according to ADI/DSAI user needs. These tools include Python libraries such as `dice-ml`, `dalex`, `erroranalysis`, `scikit-learn`, and UI visualization libraries such as `ExplainerDashboard`. This toolkit aims to: 
- Provide a user interface to guide users step-by-step in the testing process; 
- Support certain binary classification and regression models that use tabular data 
- Produce a basic summary report to help DSAI System Developers and Owners interpret test results 
- Intended to be deployable in the user’s environment

This Toolkit, along with the Guidelines document, is the first major version release. Subsequent versions and updates will be found in the XRAI [GitHub repo](https://github.com/aboitiz-data-innovation/XRAI), accompanied by quarterly XRAI Brownbag sessions.

# Introduction
## Assumptions / Limitations
The toolkit is in its second iteration
 of development. As such, there are limitations which hinders the Toolkit from being able to handle certain models and display other XRAI-related features. These include:
- V2 can only handle binary classification models and regression models 
- Only Python users for V2. An R-based Toolkit may be released according to demand and necessity for later versions 
- Certain features may be discussed in Guidelines V2 but are not yet in Toolkit V2 
- This notebook does not define ethical standards. It provides a way for DSAI System Developers and Owners to demonstrate their claims about the performance of their DSAI systems according to the XRAI principles
- Protected groups are those groups you want to compare model performance on them to all data they can be defined as this format `{"feature name 1":  value, "feature name 2":  value,...}`  you can define as many protected groups as you want, and they can be numerical or even categorical, such as `{age: 23, sex: "female"}`.


## Inputs
Our interactive toolkit only needs two main inputs before any major analysis:
- Model (.pkl or .sav)
- Data (train, test) (.csv) 

We intend for the user to have inputs mostly on the XRAI-related functions. However, we need the user to manually input **the names of train and test file**, in addition to target variable name. Prompts will be shown later in the notebook where you will need to load. In the shared folder we have provided sample model and data (test_data.csv, train_data.csv, finalized_model.pkl). For testing for different models and data you may just replace files.

## Features found in the notebook
- Fairness 
- Model Performance Overview
- Local Explainability
- Global Explainability
- Stability