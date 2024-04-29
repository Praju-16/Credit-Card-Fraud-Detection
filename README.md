# Credit-Card-Fraud-Detection
![image](https://github.com/Praju-16/Credit-Card-Fraud-Detection/assets/141834374/c05975d1-a7ef-43f4-8e32-b4f20c8cea1a)

Build a machine learning model to identify fraudulent credit card transactions.

Intern Information

Name :- Prajakta Satish Pawar

ID :- COD6991

Overview

For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.

In the banking industry, credit card fraud detection using machine learning is not just a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees, and denials of legitimate transactions.

Dataset Description

The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. 
Out of a total of 2,84,807 transactions, 492 were fraudulent. 
This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. 
The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. 
Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. 
The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. 
The feature 'amount' is the transaction amount. 
The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.

Problem statement

The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning models.
In this project, we will analyse customer-level data which has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group.
The dataset is taken from the Kaggle website and it has a total of 2,84,807 transactions, out of which 492 are fraudulent. Since the dataset is highly imbalanced, so it needs to be handled before model building.

Getting Started

To run this project locally, follow these steps:

1.	Clone the repository to your local machine:
	
https://github.com/Praju-16/Credit-Card-Fraud-Detection.git

2.Install the required Python libraries using pip:

pip install –r requirements.txt

3.Download the Titanic dataset (CSV file) from Kaggle and place it in the project directory.
	
Run the Jupyter Notebook or Python script to explore the dataset, preprocess the data, build the model, and evaluate its performance.

Evaluation

The model's performance is assessed using various metrics, including:

•	Accuracy: The proportion of correct predictions.

•	Precision: The ability of the model to correctly identify positive cases.

•	Recall: The ability of the model to capture all positive cases.

•	F1-Score: The harmonic mean of precision and recall.

•	Classification Report: A comprehensive report displaying precision, recall, and F1-Score for each class 

Output 

![image](https://github.com/Praju-16/Credit-Card-Fraud-Detection/assets/141834374/9eca77d2-2b72-43d2-a0e8-88665c8f1a82)

