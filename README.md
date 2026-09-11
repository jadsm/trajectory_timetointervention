
**Models for time to gastrostomy prediction**

This repository contains Python and R code for ALS risk modelling using the ENCALS dataset.

**January 2025**

**Publication**: Delgado-SanMartin, J. et al. Modelling Time to Gastrostomy Intervention in Amyotrophic Lateral Sclerosis: A Multi-Cohort Longitudinal Trajectory and Survival Analysis. JMIR Medical Informatics. Submitted Feb 26. Under review.<br> 
**Authors**: Delgado-SanMartin, J. (Imperial College London), Gupta V. (A*Star Singapore) <br>
This repository contains portions of other publications:
- Westeneng, H. J. et al. Prognosis for patients with amyotrophic lateral sclerosis: development and validation of a personalised prediction model. Lancet Neurol 17, 423-433 (2018). https://doi.org/10.1016/S1474-4422(18)30089-9
- Barnwal, A. C., H; Hocking, T D. Survival regression with accelerated failure time model in XGBoost.  (2020). https://doi.org/10.48550/arXiv.2006.04920

We have modified the code to suit our needs, but the authorship has been adequately cited in each code file.

# Main Files

## Trajectory analysis - GMM-DTSA
* **MPLUS_Samplescript_JointGrowthAnalysis_DiscreteSurvivalAnalysis.inp** (input file): GMM-DTSA model configuration

## Gastrostomy prediction
* **preprocess_datasets.py** (Python Script): preprocesses datasets for use in the modelling pipeline.
* **hyperpar_opt_right_censored.py** (Python Script): trains optimizes hyperparameters for all right-censored survival models.
* **train_right_censored.py** (Python Script): trains a right-censored survival model on the preprocessed datasets. Note: this and hyperpar_opt_right_censored.py both help you train the model, this one being more performant and thorough in the model reporting metrics.
* **utils** (Python Folder): utility functions used throughout the project. Utils is broken down into: 
    - **constants.py** (contains constant parameters), 
    - **utils_censor.py** (censoring utility functions), 
    - **utils.py** (general utilities) and 
    - **parser_utils.py** (utils for the parser).
* **plot_right_censored.py** (Python Script): generates model result plots for right-censored survival data
* **simulation.py** (Python Script): simulates data from the model to assess model generalizability and analyse sensitivity. It computes other meta-analyses reflected in the manuscript.

**Dependencies**

This codebase likely uses libraries for data manipulation, statistical analysis, and potentially deep learning. The specific libraries can be determined by examining the code itself.

# Getting Started

1. Install required libraries (refer to the code for specific requirements).
2. Ensure you have R and a suitable R environment set up if the  **ENCALS_risk_modelling.R** script is used.
3. Run the scripts in the appropriate order to perform the data preprocessing, model training, evaluation, and visualization.

# Further Notes

* This README provides a general overview of the files based on their naming conventions. Refer to the code itself for specific details about the functionality.
* For questions on trajectory analysis, please contact Dr Varsha Gupta (varsha_gupta@sics.a-star.edu.sg) and for anything else Dr Juan Delgado (j.delgado-san-martin@imperial.ac.uk)


