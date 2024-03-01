# Cancer Patient Data Analysis and Classification

## Overview
This project aims to analyze cancer patient data and develop classification models to predict the severity of cancer. The dataset used for this analysis contains various attributes related to cancer patients, including demographic information and medical indicators. By leveraging machine learning and deep learning techniques, we seek to build accurate models that can assist in diagnosing and treating cancer patients effectively.

## Dataset
The dataset (`cancer_patient_data_sets.csv`) consists of the following attributes:
- `Patient Id`: Unique identifier for each patient
- `Age`: Age of the patient
- `Gender`: Gender of the patient (male/female)
- `Air Pollution`: Level of air pollution in the patient's environment
- `Alcohol use`: Frequency of alcohol consumption
- `Dust Allergy`: Presence of dust allergy (yes/no)
- `OccuPational Hazards`: Exposure to occupational hazards (yes/no)
- `Genetic Risk`: Genetic predisposition to cancer (yes/no)
- `chronic Lung Disease`: Presence of chronic lung disease (yes/no)
- `Balanced Diet`: Adherence to a balanced diet (yes/no)
- `Obesity`: Obesity status (yes/no)
- `Smoking`: Smoking habits (yes/no)
- `Passive Smoker`: Exposure to passive smoking (yes/no)
- `Chest Pain`: Occurrence of chest pain
- `Coughing of Blood`: Presence of blood in cough
- `Fatigue`: Fatigue level
- `Weight Loss`: Unexplained weight loss
- `Shortness of Breath`: Experience of shortness of breath
- `Wheezing`: Wheezing symptoms
- `Swallowing Difficulty`: Difficulty in swallowing
- `Clubbing of Finger Nails`: Clubbing of finger nails
- `Frequent Cold`: Frequency of colds
- `Dry Cough`: Presence of dry cough
- `Snoring`: Snoring habits
- `Level`: Severity level of cancer (target variable)

## Dependencies
Ensure you have the following dependencies installed:
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- tensorflow

You can install the dependencies using pip:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib tensorflow
```



## Analysis Steps
1. **Data Loading and Preprocessing:**
   - Load the dataset using pandas.
   - Drop unnecessary columns (`Patient Id` and `index`).
   - Encode categorical labels using LabelEncoder.
   - Handle missing values (if any).

2. **Exploratory Data Analysis (EDA):**
   - Visualize the distribution of features.
   - Examine the correlation between features using heatmap and correlation matrix.

3. **Data Visualization:**
   - Plot histograms for feature distributions.
   - Visualize boxplots for outlier detection.

4. **Data Splitting:**
   - Split the dataset into training and testing sets using `train_test_split` from scikit-learn.

5. **Model Building and Evaluation:**
   - Train and evaluate three classification models:
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Logistic Regression
   - Evaluate model performance using accuracy metrics.

6. **Neural Network Model:**
   - Define and train a neural network model using TensorFlow.
   - Compile the model with appropriate loss function and optimizer.
   - Train the model and visualize training history.

## Files
- `cancer_patient_data_sets.csv`: Dataset containing cancer patient data.
- `cancer_classification.py`: Python script for data analysis and classification.

## Contributors
Abdelrahman Mohamed https://github.com/AbdelrahmanBedo

##DATA 
https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/code
