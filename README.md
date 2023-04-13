![pic](https://user-images.githubusercontent.com/100692852/231856693-e44577fd-aeaf-4d5d-888c-aa5091b19428.jpg)
* Contribution: Jingcheng Chu, Utkarsh Mehta, Qingyang Fu
## Introduction
Lending Club https://www.lendingclub.com/ is a peer-to-peer (P2P) lender operating in North America, known for its innovative business model of mitigating risk by diversifying investments across multiple loans. In a typical loan scenario, borrowers receive a sum of money, or principal, and are required to repay the lender this principal along with interest in the future.

However, lending involves credit risk, which is the potential for a borrower to default on a loan by failing to make the necessary payments. This risk exposes the lender to the loss of principal, interest, and disrupted cash flows. As a result, assessing the credit risk associated with a loan portfolio is crucial, as it allows for the prediction of potential borrower default, enabling the lender to take preventative action and maximize profits.
In this project, we have developed a fully compliant advanced Internal Ratings-Based (IRB) model for origination scoring using data collected between 2007 and 2018. By leveraging historical credit information on existing borrowers and their interactions with the bank, we have created a Probability of Default model that accurately evaluates the risk associated with each borrower.

### 1.1 Data Description
Link to the lending club- Lending Club.
For our analysis, we utilized dLending Club's loan data for between 2007 and 2018. The dataset initially consisted of 1,280,646 rows and 128 columns. Within these 128 columns, 38 were classified as Performance Metrics, while the remaining 90 were categorized as Original Variables. Before delving into the analysis, it was important to address data irregularities, such as missing values and outliers, which would be discussed in the subsequent sections.
## Methodology

### 2.1 Data Cleaning
Our data cleaning process consisted of two distinct stages, focusing on the Origination and Performance datasets independently. We adhered to a comprehensive set of rules to ensure data integrity while preserving all valuable observations. Additionally, we assessed each variable on an individual basis to determine whether missing data was random or attributable to specific causes, as outlined in Table 2.1.1.

Firstly, we addressed completely empty observations. For example, our dataset contained 31 observations with null values across all columns, which we then removed. Next, we conducted a case-by-case analysis for both the Origination and Performance datasets, implementing three primary strategies: conversion, imputation, and removal.

For the Origination dataset, we applied imputation techniques to observations with missing values to maintain data consistency and accuracy. We substituted missing values with the median, as it was less susceptible to outliers. In exceptional cases, we replaced missing values with -1 or 0, depending on the underlying cause. For instance, we filled in 0 for the employment length of unemployed individuals with missing values, for variables related to months, such as 'mths_since_recent_bc_dlq', we used -1. These variables indicated the number of months since the most recent delinquency records were published, and we assigned a -1 to individuals without public records to preserve data accuracy.

Furthermore, we converted unique values to categorical values, enabling further analysis of variables like term and employment length. We also created new columns to merge duplicate data, such as combining annual income and joint annual income into a single column, combined annual income. We consolidated the loan statuses of charge off and default as our default status, treating all other statuses as ongoing payments.

Lastly, we eliminated redundant or irrelevant variables, such as those related to secondary applications. We also discarded variables with an excessive number of unique and less informative values, like address and zip code.

In the Performance dataset, we followed the same procedures as in the Origination dataset. We removed 31 entirely empty datasets, converted loan status to 0 and 1 to represent default and non-default, respectively, and removed variables that were redundant or irrelevant to our LGD and PD estimates and predictions, as illustrated in Table 2.1.2.

### 2.2 Design Ratios
In our project, we designed three financial ratios to improve the predictive power of our model: the Payment-to-Income (PTI) ratio, the Revolving Balance Ratio (RBR), and the Debt-to-Limit (DTL) ratio. The PTI ratio allowed us to evaluate an individual's capacity to manage debt repayment by dividing the total loan payments by their annual income. We found that a lower PTI ratio made borrowers more attractive to lenders, while a high ratio suggested that the borrower might struggle to meet their debt obligations.

The RBR, also known as the credit utilization ratio, measured the proportion of available credit currently in use. It provided insights into a borrower's creditworthiness, with a low ratio indicating responsible credit management and a high ratio indicating possible overextension.

Lastly, the DTL ratio excluded mortgage balances and compared the proportion of non-mortgage debt to the total available credit. This metric gave us insight into a borrower's creditworthiness and their responsible use of available credit. We found that a lower DTL ratio was considered favorable, while a higher ratio might suggest that the borrower was overextended and could have difficulty managing their debt.

### 2.3 Outliers
Dealing with outliers is essential in data analysis, as they can have a significant impact on the results of our analysis, potentially leading to erroneous conclusions. We generated KDE (Kernel Density Estimation) plots to inspect individual variables and determine which ones had outliers. In total, there were outliers present in 44 variables.

To deal with extremely skewed variable data and outliers, we "winsorized" the data. We replaced the extreme values that were above the 75% percentile with the value at the 75% quantile. We adopted this approach to reduce the impact of extreme values on the analysis while still preserving the overall distribution of the data. Winsorizing improved our model performance and reduced the distortion of statistical measures, making our results more reliable and interpretable.

### 2.4 WoE Transformation
WOE refers to the Weight of Evidence Transformation and is a method used to transform categorical and continuous variables into continuous variables that can be used as inputs for machine learning models. The basic idea was to transform a categorical variables into new variables, expressed as “the weight of evidence” for each category. This helped us capture the relationship between the variables and the outcome of interest (default and non-default) by comparing the distribution of each category in the target group.

To implement WoE transformation, we split data into train and test sets, binned the variable, and filtered out low predictive power variables using IV scores. Using 0.02 IV as a strict cutoff and 0.1 as a lenient cutoff, we removed 29 columns. (refer appendix Table 2.4.1). 

We manually adjusted binning in the remaining variables to improve plot trends. Only a few columns needed manual binning, and their IV scores decreased further. We removed six more columns with IV scores smaller than 0.02 (refer appendix Table 2.4.2). Then we applied the adjusted breaks to both train and test data, and plots looked good, enabling further analysis.

### 2.5. Correlation

For the remaining variables, it was crucial to examine their correlation. To do this, we constructed a correlation plot(refer appendix Figure 2.5.2). When two variables are correlated, changes in one variable corresponds with changes in the other, which can result in redundancy. Redundant variables do not contribute additional information to the analysis, and removing them can simplify the process and improve interpretability. Our approach involved identifying pairs of correlated variables and retaining the variable with the higher Information Value (IV) score while dropping the one with the lower score. This method allowed us to eliminate 14 variables while preserving those with better predictive capabilities (refer appendix Table 2.5.1). 

## Modeling

### 3.1 Logistic Regression

Logistic regression is a type of regression analysis that is used to model the relationship between a binary dependent variable and one or more independent variables. The logistic regression model is based on the logistic function, which maps any real-valued input to a value between 0 and 1, representing the probability of the dependent variable taking on the value of 1.

In our Credit Risk Analytics, the dependent variable was the default status and we had 24 independent variables. We used the WoE (weights of evidence) transformed dataset for the purpose of the logistic regression. We modified several logistic regression parameters according to the needs of our data set. For details on parameter values used, refer to appendix Table 3.1.

After creating the model, we fitted it with the train_woe dataset. In order to verify our model, we predicted the Probability of Default (PD) using the test_woe. Next, we constructed a confusion matrix (Figure 3.1) to understand the performance of our logistic model. We used the predicted classes of default and the actual true classes from default status to construct a confusion matrix. The obtained confusion matrix is represented below:
![image](https://user-images.githubusercontent.com/100692852/231858963-f1393d5d-257b-4543-81b3-9f01e12fb7ae.png)


Explanation for Confusion Matrix:

- The actual positive and predicted positive value is 0.67. This implies that our model is able to accurately predict 67% of defaulters. 
- The actual negative and predicted negative value is 0.63. This implies that our model is able to accurately classify 63% of non-defaulters.
- The actual positive and predicted negative value 0.33 which means that our model is wrongly classifying 33% of non-defaulters as defaulters. 

We get an Accuracy score of 0.65 or 65% accurate logistic model.

We had a highly imbalanced dataset with more non-defaulters than defaulters. Considering that we were able to correctly classify more than 60% of defaulters and non-defaulters, we can continue with further analysis. 

### 3.2 ScoreCard

A scorecard as shown in Figure 3.2  is a statistical model that is used to evaluate the creditworthiness of borrowers based on their credit history and other relevant factors. The scorecard assigns a score to each borrower that represents their likelihood of defaulting on a loan or credit card payment.

![image](https://user-images.githubusercontent.com/100692852/231854430-7a0136f8-cf73-41b4-af45-06f23662896d.png)

This implies that we have successfully created a scorecard for 896430 customers on our training dataset and we achieve a minimum score of 258, a maximum score of 680 and a mean score of 431.49. 

### 3.3 XGBoosting

XGBoost is a widely-used machine learning algorithm, renowned for its ability to efficiently handle large datasets containing various data types. We selected this method as an alternative to predicting default status, comparing its results with those generated by Logistic Regression in the previous section.

XGBoost performs classification and regression by utilizing a series of weak decision trees, each trained on the residual errors (i.e., the difference between predicted and actual values) of the preceding tree. Each tree is designed to correct the errors made by the previous one. The final prediction of the model is formed by combining the predictions of all trees in the sequence.

We began by creating a baseline model using default parameters, which we subsequently refined through further analysis. To identify the optimal combination of hyperparameters for our XGBoost model, we constructed a parameter grid to be used with Grid Search. This grid included three hyperparameters: learning rate, number of estimators, and max depth of each decision tree in the ensemble. The learning rate determines the weight updates for each iteration, while the number of estimators indicates how many decision trees the model should use.

We employed Grid Search, a parameter tuning technique, to discover the best combination of hyperparameters. Due to the size of our dataset, we fitted Grid Search using a randomly selected 50% subset of our training set. The optimal hyperparameters selected were: 1000 estimators, a max depth of 4, and a learning rate of 0.05. Using these tuned hyperparameters, we constructed our optimal XGBoost model.

To fit the data, we established a pipeline for our Grid Search and XGBoost model, comprising a preprocessor and the model itself. Before fitting each model, we applied the preprocessor, which utilized one-hot encoding to convert categorical data into numerical format. After fitting our final model, we tested its predictive ability on the test set. However, the results were less than ideal. Our trained XGBoost model exhibited a high false-positive rate of 86% and an accuracy of 55% (Figure 3.3). Due to its suboptimal predictive ability, we opted for Logistic Regression to predict the Probability of Default (PD) in our subsequent analysis. We concluded that the unsatisfactory performance of the XGBoost model could be attributed to our imbalanced dataset and potential overfitting in the model.

![image](https://user-images.githubusercontent.com/100692852/231854490-3751f9a4-6e69-4a5b-a3ae-7514e8ad2795.png)

### 3.4 Variable Importance

We constructed two predictive models using Logistic Regression and XGBoost and evaluated their variable importance using distinct methods after training them.

For the XGBoost model, we computed the SHAP (SHapley Additive exPlanations) values using the SHAP explainer. Subsequently, we plotted the variable importance based on the SHAP values, sorting them in descending order. As evident from Figure 3.4.1 in Appendix, the remaining installment and remaining amount funded by investors were the most significant features in the XGBoost model.

![image](https://user-images.githubusercontent.com/100692852/231855622-0a557daa-a053-470d-98a8-31f107022e59.png)


In the case of the Logistic Regression model, we assessed variable importance based on the magnitude of their coefficients. The loan term and the total high credit limit were the most significant features in the Logistic Regression model (Table3.4.2 Appendix).

![image](https://user-images.githubusercontent.com/100692852/231855526-202151c9-3de7-4b6e-af19-9e8671dab8f8.png)


Upon comparing the variable importance of the two models, we observed that several features were significant in both models, although there were also notable differences in the significant variables. This can be attributed to the fact that the two models were trained on different datasets: Logistic Regression was trained on the train-woe dataset, containing 23 predictors, while XGBoost was trained on the train dataset, containing 68 predictors.
 

## Result
### 4.1 Optimal Cutoff Point

The Optimal Cutoff Point is a critical threshold used to assess loan eligibility based on an individual's Probability of Default (PD) score. If a person's PD score exceeds the cutoff point, they are deemed ineligible for the loan; otherwise, they are considered eligible. The cutoff point acts as a decision-making rule for approving or denying loans based on an applicant's PD score.

Our objective was to maximize profits for the lending club by identifying the Optimal Cutoff Point, which would enable us to extend loans to the largest number of applicants while minimizing default rates. To determine this cutoff point, we employed metrics to estimate profit, such as Loss Given Default (LGD) and Exposure at Default (EAD).

Firstly, we applied Logistic Regression to calculate the Probability of Default (PD), as it demonstrated higher accuracy than the XGBoosting model. We used the entire dataset for PD prediction after implementing the WoE transformation.

Subsequently, we computed the EAD by deducting the total principal received from the total funded amount, yielding the total amount at risk of default. We also estimated the LGD as 1 minus the recovery rate, with the recovery rate calculated as the total recovered amount divided by the EAD. Multiplying EAD by LGD provided us with the loss at the event of default, representing the cost of a loan in case of a default.

To determine revenue, we took the installment interest rate minus the cost of funds, multiplied by the installment value, and discounted this over the remaining payments using a suitable discount rate. This calculation reflected the lending club's gain when a loan was repaid.

Table 4.1 illustrates the total profit calculations at various cutoff points, enabling us to identify the optimal cutoff point that maximized profit. We determined the optimal cutoff point to be a PD of 0.45, which permitted us to accommodate a broader pool of potential borrowers while maintaining default risks at acceptable levels. This approach ultimately led to increased profits for the lending club.


![image](https://user-images.githubusercontent.com/100692852/231854660-2df3e176-1050-435d-8004-abc05181f64f.png)


### 4.2 PD Calibration

To calculate the monthly PD, we first needed to derive PD segments by segmenting the Receiver Operating Characteristic (ROC) curve of our initially computed PD from Logistic Regression. We then fitted the best curve using a piecewise Area Under the Curve (AUC) method. This involved dividing the range of predicted PD into segments and determining the AUC for each segment individually (Appendix Figure 4.2.1.)


![image](https://user-images.githubusercontent.com/100692852/231854696-0b1aa7b7-5a0e-427a-81a1-dd5cf85cf61f.png)

After computing the cuts, we applied them to our dataset. As depicted in Figure 4.2.2, the piecewise curve closely matched our original ROC curve. We also ensured that the cuts resulted in monotonic PDs by calculating the PD for the entire dataset (Appendix Table 4.2.3). The monthly PDs were monotonous, indicating that our cuts were appropriate for the data. Upon examining the cuts, we combined the first two cuts since the first cut did not contain any defaulters.

![image](https://user-images.githubusercontent.com/100692852/231854730-2bfd5dd8-0a21-4fa9-89eb-825479fea7b8.png)

![image](https://user-images.githubusercontent.com/100692852/231854781-6593be5c-8d59-4d66-9653-ed43f0bc88b4.png)

Next, we calculated the PDs for each portfolio by determining the average number of defaults for each portfolio, divided by the total number of cases in the month the loan was issued. We converted the loan issue dates into numerical values ranging from 1 to 138, representing months from July 2007 to December 2018 (Appendix Figure 4.2.4).

![image](https://user-images.githubusercontent.com/100692852/231854817-5f620cb7-b8c8-4e4b-927d-d28cf4b35276.png)

### 4.3 Time Series Analysis

After obtaining monthly PD values, we employed Time Series Analysis techniques to forecast a long-run PD. Specifically, we used SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) to model the data.

To incorporate relevant macroeconomic factors into the model, we obtained long-term forecast data on key economic indicators such as unemployment rate, interest rate, and M1 rate from the website of Bank of St. Louis (link in references). These factors were used as exogenous variables in our SARIMAX model.

Prior to fitting the SARIMAX model involving the macroeconomics factors, we decomposed our monthly PD to examine its trend. As shown in Figure 4.3.1 the time series of monthly PD were stationary, with yearly seasonality, indicating that it was ready for model fitting. 

We created seasonal parameters for the SARIMAX model to tune, and we picked the best set of seasonal parameters for our final model, based on AIC (Akaike Information Criterion), and the best set of parameters was (3, 1, 0), (0, 0, 0, 12). We then fitted our best SARIMAX model using the tuned seasonal parameters which produced the output like we showed.

Based on the model output Figure 4.3.2, we found that the exogenous information provided by the macroeconomic factors did not significantly improve the model's accuracy. Therefore, we determined that additional relevant economic factors were necessary to further calibrate our long-run PD forecast.

![image](https://user-images.githubusercontent.com/100692852/231854865-3b203921-8e33-42e1-8f36-b58f11b911a4.png)

![image](https://user-images.githubusercontent.com/100692852/231854890-1550bcd3-210d-481b-9034-39d7a62cf034.png)

## Conclusion

Within this study, we have developed a fully compliant advanced IRB model for origination scoring using the Lending Club dataset. Our cleaning process ensured that the data was in a usable format for our models. We also designed three new variables that helped improve the accuracy of our models. Our constructed scorecard demonstrated stronger performance compared to the XGBoosting. Variable importance analysis provided us with useful insights into the factors that influence the risk of default.

We also calculated the optimal cutoff point for PD, which helps Lending Club maximize their profits while managing their risk. Moreover, we attempted to calibrate a long-term PD model by incorporating external macroeconomic variables, but our findings were not statistically significant.

In conclusion, our analysis of the Lending Club dataset provides valuable insights for lenders and investors interested in managing risk associated with peer-to-peer lending. Our predictive models can help Lending Club make informed decisions about lending, while also allowing regulators to estimate capital requirements for the company if they were to be regulated.


## REFERENCES
Federal Reserve Bank of St. Louis. (n.d.). Federal Reserve Economic Data: Fred: St. louis fed. FRED. Retrieved April 10, 2023, from https://fred.stlouisfed.org/ 


Oecd. (n.d.). OECD statistics. OECD Statistics. Retrieved April 10, 2023, from https://stats.oecd.org/ 


## APPENDIX: Tables and Figures


![image](https://user-images.githubusercontent.com/100692852/231858028-5aa66971-8d75-4079-b391-32dfcf00d79d.png)

![image](https://user-images.githubusercontent.com/100692852/231858193-c5b6be18-7473-4318-a692-9dfb7d73fa27.png)

![image](https://user-images.githubusercontent.com/100692852/231858548-bddacea2-dc43-422a-a10b-79f458d990b8.png)

![image](https://user-images.githubusercontent.com/100692852/231858232-063d80b2-3f71-4a70-8fb6-e17666e4cf8e.png)

![image](https://user-images.githubusercontent.com/100692852/231858279-884dd9fb-2d39-4cac-a46f-d0da02b4a8cb.png)

![image](https://user-images.githubusercontent.com/100692852/231858316-862e22f5-e4ec-4cbe-ab8f-2d8eeb18512b.png)

![image](https://user-images.githubusercontent.com/100692852/231858353-4ede0a66-4bf2-4831-a34f-fbc0cc1b616e.png)

![image](https://user-images.githubusercontent.com/100692852/231858371-f9550643-2778-4234-8814-eabf82c65dde.png)

![image](https://user-images.githubusercontent.com/100692852/231858393-7987f7af-5b0c-43ed-a887-3b0d571ef52d.png)


