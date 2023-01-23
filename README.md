# Practical Application 3

In this third practical application assignment, the goal is to compare the performance of the classifiers:
* k-nearest neighbors
* logistic regression
* decision trees
* support vector machines

## Data

The [dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) we are going to use comes from the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/index.html) 

__Created by__: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012

__The full dataset was described and analyzed in__:

[Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.

### Data Description

The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns. 

The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. 

__There are 4 datasets__:

1. __bank-full.csv__ with all examples, ordered by date (from May 2008 to November 2010).
2. __bank.csv__ with 10% of the examples (4521), randomly selected from bank-full.csv.
3. __bank-additional-full.csv__ with all examples, ordered by date (from May 2008 to November 2010).
4. __bank-additional.csv__ with 10% of the examples (4119), randomly selected from bank-additional-full.csv.

The data is enriched in (3) and (4) by the addition of five new social and economic features/attributes (national wide indicators from a ~10M population country), published by the Banco de Portugal and publicly available at: https://www.bportugal.pt/estatisticasweb.
   
(2) and (4) dataset is provided to test more computationally demanding machine learning algorithms (e.g., SVM).

__We will be using (1) and (2) (specifically for SVM, as you will see) in this exercise.__

__The binary classification goal is to predict if the client will subscribe a bank term deposit (variable y).__

#### bank client data:
- age (numeric)
- job : type of job ("admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services") 
- marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
- education (categorical: "unknown","secondary","primary","tertiary")
- default: has credit in default? (binary: "yes","no")
- balance: average yearly balance, in euros (numeric) 
- housing: has housing loan? (binary: "yes","no")
- loan: has personal loan? (binary: "yes","no")

#### related with the last contact of the current campaign:
- contact: contact communication type (categorical: "unknown","telephone","cellular") 
- day: last contact day of the month (numeric)
- month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
- duration: last contact duration, in seconds (numeric)

#### Other attributes:
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -means client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client (numeric)
- poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

#### Output variable (desired target):
- y - has the client subscribed a term deposit? (binary: "yes","no")

#### Missing Attribute Values
There are no missing attributes.

## Goals

__The goal of this exercise is to build a data-driven prediction system to help banking firms achieve a high degree of success with telemarketing efforts for their banking products, such as `fixed term deposit`.__

## EDA (Exploratory Data Analysis)

### DataFrame analysis
- There are 16 features
- One field for binary classification
- No nulls
- 45211 rows of data, which we will split into 70/30, train/test sets

__Let's look closely at the various feature columns next__

__9 of the features are non-numeric. Looking at the value counts:__
- job, marital, education, contact, poutcome (5) are non-ordinal categorical columns
- housing, loan and default (3) are binary categorical columns 
- month is an ordinal categorical column

__We will employ 1hot encoder for the non-ordinal and binary categorical columns and ordinal encoder for month.__

```
Unique values in Categorical Column job:
blue-collar      9732
management       9458
technician       7597
admin.           5171
services         4154
retired          2264
self-employed    1579
entrepreneur     1487
unemployed       1303
housemaid        1240
student           938
unknown           288
Name: job, dtype: int64
---------------------------------------------
Unique values in Categorical Column marital:
married     27214
single      12790
divorced     5207
Name: marital, dtype: int64
---------------------------------------------
Unique values in Categorical Column education:
secondary    23202
tertiary     13301
primary       6851
unknown       1857
Name: education, dtype: int64
---------------------------------------------
Unique values in Categorical Column default:
no     44396
yes      815
Name: default, dtype: int64
---------------------------------------------
Unique values in Categorical Column housing:
yes    25130
no     20081
Name: housing, dtype: int64
---------------------------------------------
Unique values in Categorical Column loan:
no     37967
yes     7244
Name: loan, dtype: int64
---------------------------------------------
Unique values in Categorical Column contact:
cellular     29285
unknown      13020
telephone     2906
Name: contact, dtype: int64
---------------------------------------------
Unique values in Categorical Column month:
may    13766
jul     6895
aug     6247
jun     5341
nov     3970
apr     2932
feb     2649
jan     1403
oct      738
sep      579
mar      477
dec      214
Name: month, dtype: int64
---------------------------------------------
Unique values in Categorical Column poutcome:
unknown    36959
failure     4901
other       1840
success     1511
Name: poutcome, dtype: int64
---------------------------------------------
Unique values in Categorical Column y:
no     39922
yes     5289
Name: y, dtype: int64
---------------------------------------------
```

For the numerical columns, there does not seem to be any strong correlation. Which implies we don't need to eliminate any columns and can use all of them. 

__Previous and pdays have some weak correlation. Previous being the number of minutes spent previously with the customer trying to sell them on the subscription and pdays being the number of days since last contact.__

__Most notably the prediction label is binary but imbalanced:__
- `yes` is 11.7%
- `no` is 88.3%

![Correlation Matrix](images/corr.png "Correlation Matrix")

![Pair Plot](images/pairplot.png "Correlation Matrix")

### Individual features and their relation to the prediction label

- __`yes` values seem to be distributed more widely across age brackets__
![age](images/newplot-1.png "Campaign Subscription broken out by age")
- __`bluecollar`, `management` and `technical` represent the top 3 candidates for outreach__
![job](images/newplot-2.png "Campaign Subscription broken out by Job type")
- __`married` people get the most outreach but also say `no` more often__
![marital](images/newplot-3.png "Campaign Subscription broken out by Marital Status")
- __`secondary` education level gets the most calls but seems to have a lower rate of acceptance than `tertiary`__
![education](images/newplot-4.png "Campaign Subscription broken out by education level")
- __Folks who have not defaulted in the past get the majority of the calls, indicating a preference for lower risk__
![default](images/newplot-5.png "Campaign Subscription broken out by whether the customer has Defaulted")
- __Most people getting outreach are in the low bank balances__
![balance](images/newplot-6.png "Campaign Subscription broken out by Bank Balance")
- __No clear patterns but people who say `yes` tend to be in the lower income brackets, as seen earlier, and across a wide age range.__ That said, bank balances of people who say yes tends to be slightly higher and age tends to be a little lower (based on the box plots). There is also a correlation between age and balance albeit a weak one. 
![balancevage](images/newplot-7.png "Age vs. Balance and Subscription Status")
- __Those without `housing` seem to say `yes` at a higher rate.__ 
![housing](images/newplot-8.png "Campaign Subscription broken out by housing")
- __Majority of folks who get calls don't have loans__
![loan](images/newplot-9.png "Campaign Subscription broken out by loan")
- __Most people are contacted over their cellphones and also have a high acceptance rate__
![contact](images/newplot-10.png "Campaign Subscription broken out by Contact type")
- __No clean patern on days of the months.__
![day](images/newplot-11.png "Subscription by day of month")
- __A majority of outreach happens in May for some reason. Budgetary?? Holiday season sees lower volumes as one would expect__
![month](images/newplot-12.png "Subscription by month")
- __Most people who said `no`, said so pretty quickly. Those saying `yes` took more contact duration__
![](images/newplot-13.png "Subscription by duration of contact during last pitch")
- __Most folks who were contacted were never contacted before (although there are quite a few `-5` values and it's unclear why that is different than `-1`). Filtering these out, most people who said were last contacted less than a year ago__
![contact](images/newplot-14.png "Subscription by Days since Past Contact")
- __There is one outlier who was contacted >200 times (could be an error incorrect data) and skews the visualization. Eliminating this outlier, we see that most folks who said yes were 30 or less times__
![previousvpdays](images/newplot-15.png "Days Since Past Contact vs. No. of Previous Contact and Subscription Status")

### Data Summary

we have 16 input features from which we can train a model to predict whether a customer signed up for a `fixed term deposit` program when they were offered that program. 

Input features come in various flavors:

- Customer attributes such as their age, bank balance, marital status or loans
- behavior/operational attributes such as when they were contacted last, how long we spoke to them about the `fixed term deposit` program, number of times we have already contacted them etc. 

No null values or significant outliers exist. Neither is there a string covariance between any two input features.

Data does have quite a few categorical columns which will need to be encoded appropriately and data of different scales (like age or bank balance) and hence will have to be scaled. 

## Model Training

### Column transformer we will use for all models

Copying from before, of all the categorical features:

- 9 of the features are non-numeric. Looking at the value counts:
- job, marital, education, contact, poutcome (5) are non-ordinal categorical columns
- housing, loan and default (3) are binary categorical columns
- month is an ordinal categorical column

__In addition we will scale the  features in a pipeline,. This is important since there are several features with different numerical scales (for e.g. 1hot encoded features vs. age vs. balance)__

### Model training

In the below section we will train the following models:

- K Nearest Neighbors
- Logistic Regression
- Decision Trees
- Support Vector Machines

For each model, we will use the above column transformer in a pipeline which also includes data scaling after transforming the features. 

__We will not be doing any PCA or feature selection  steps since the [paper](https://www.sciencedirect.com/science/article/abs/pii/S016792361400061X) alludes__ to the fact that a semi automatic feature selection was employed to get to the list of features in the provided dataset. 

__We will also not do employ polynomial transformers for any features__

For each model we will employ a __grid search__ as follows:

- KNN: `n_neighbors`, `p` and `weights`
- Logistic Regression: `penalty` and since we have imbalanced classes (`yes` is 11.7%, `no` is 88.3%) we will set `class_weight` as `balanced`
- Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leag` and `criterion`
- SVM: `degree`, `kernel` and since we have imbalanced classes (`yes` is 11.7%, `no` is 88.3%) we will set `class_weight` as `balanced`

We will also time the models using the [Jupyter Notebook `%timeit` magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

#### Scoring 
__When comparing the models will be `ROC_AUC`__ just like the authors of the [paper](https://www.sciencedirect.com/science/article/abs/pii/S016792361400061X), and this makes sense since with a business goal such as this (`telemarketing banking products to potential customers`), we don't want to spam our users (precision) but a the same time we want to maximize revenue potential and not miss any customers who might be interested (recall). 

#### KNN

The model took `9min 39s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)` to grid search and train.

```
Best parameters for KNN found during grid search={'knn__n_neighbors': 9, 'knn__p': 2, 'knn__weights': 'distance'}
              precision    recall  f1-score   support

          no       0.91      0.98      0.94      9989
         yes       0.60      0.28      0.38      1314

    accuracy                           0.89     11303
   macro avg       0.76      0.63      0.66     11303
weighted avg       0.88      0.89      0.88     11303

ROC_AUC Score for best KNN = 0.8288306254078879
False Positive Rates:  [0.         0.00030033 0.00030033 0.00050055 0.00050055]
True Positive Rates:  [0.         0.00684932 0.01369863 0.01369863 0.02054795]
```
![knn-roc](images/knn-roc.png "KNN ROC AUC")
![knn-confusion](images/knn-confusion.png "KNN Confusion Matrix")

#### Logistic Regression

The model took `1.95 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)` to grid search and train.

```
Best parameters for Logistic Regression found during grid search={'lgr__penalty': 'l2'}
              precision    recall  f1-score   support

          no       0.97      0.84      0.90      9989
         yes       0.39      0.78      0.52      1314

    accuracy                           0.83     11303
   macro avg       0.68      0.81      0.71     11303
weighted avg       0.90      0.83      0.85     11303

ROC_AUC Score for best Logistic Regressor = 0.8924306082200314
False Positive Rates:  [0.         0.00010011 0.00010011 0.00020022 0.00020022]
True Positive Rates:  [0.         0.         0.00152207 0.00152207 0.00228311]
```
![lgr-roc](images/lgr-roc.png "Logistic Regression ROC AUC")
![lgr-confusion](images/lgr-confusion.png "Logistic Regression Confusion Matrix")

#### Decision Trees

The model took `25min 8s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)` to grid search and train.

```
Best parameters for Decision Tree found during grid search={'dtree__criterion': 'entropy', 'dtree__max_depth': 9, 'dtree__min_samples_leaf': 20, 'dtree__min_samples_split': 11}
              precision    recall  f1-score   support

          no       0.93      0.96      0.94      9989
         yes       0.60      0.43      0.50      1314

    accuracy                           0.90     11303
   macro avg       0.76      0.70      0.72     11303
weighted avg       0.89      0.90      0.89     11303

ROC_AUC Score for best Logistic Regressor = 0.9003741253887648
False Positive Rates:  [0.         0.00070077 0.00080088 0.00130143 0.00130143]
True Positive Rates:  [0.         0.01598174 0.02054795 0.02739726 0.03652968]
```
![dtree-roc](images/dtree-roc.png "Decision Tree ROC AUC")
![dtree-confusion](images/dtree-confusion.png "Decision Tree Confusion Matrix")

#### Support Vector Machine

The model took `4h 16min 59s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)` to grid search and train.

```
Best parameters for SVM found during grid search={'svc__degree': 1, 'svc__kernel': 'rbf'}
              precision    recall  f1-score   support

          no       0.97      0.83      0.89      9989
         yes       0.38      0.82      0.52      1314

    accuracy                           0.83     11303
   macro avg       0.68      0.82      0.71     11303
weighted avg       0.90      0.83      0.85     11303

ROC_AUC Score for best SVM = 0.8973025579278758
False Positive Rates:  [0.         0.00010011 0.00010011 0.00020022 0.00020022]
True Positive Rates:  [0.         0.         0.00076104 0.00076104 0.00152207]
```
![svm-roc](images/svm-full-roc.png "SVM ROC AUC")
![svm-confusion](images/svm-full-confusion.png "SVM Confusion Matrix")

#### SVM with Small Dataset

whoa! 4 hours to train an SVM on the full banking data. No wonder the authors also provided a [smaller data set](https://archive.ics.uci.edu/ml/datasets/bank+marketing#) which is a randomly selected 10% of the original  for both the regular feature set and the extended feature set.

Let's try with that and see if there is a huge difference in scores.

The model took `2min 22s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)` to grid search and train.

```
Best parameters for small dataset SVM found during grid search={'svc__degree': 1, 'svc__kernel': 'rbf'}
              precision    recall  f1-score   support

          no       0.97      0.83      0.90      1006
         yes       0.37      0.80      0.51       125

    accuracy                           0.83      1131
   macro avg       0.67      0.82      0.70      1131
weighted avg       0.90      0.83      0.85      1131

ROC_AUC Score for best small dataset SVM = 0.8890099403578529
False Positive Rates:  [0.         0.         0.         0.00198807 0.00198807]
True Positive Rates:  [0.    0.008 0.024 0.024 0.048]
```
![svm-small-roc](images/svm-small-roc.png "SVM ROC AUC")
![svm-small-confusion](images/svm-small-confusion.png "SVM Confusion Matrix")

__While the model trained on the full data set has a slightly higher AUC_ROC score (0.897 vs. 0.889), the resource needs are vastly different (4 hrs vs 2 mins) to train the SVM on the smaller (10%) dataset!__

## Model Comparison
![final_results](images/results.png "Final Results")

## Conclusions

Based on the above gridsearches on various models, let's compare model architectures. 

__Remember, our goal is to "build a data-driven prediction system to help banking firms achieve a high degree of success with telemarketing efforts for their banking products, such as fixed term deposit" as we listed out before.__

But before we compare, a few observations:

- Classes are imbalanced. The positive class `yes` is 11.7%. Looking at the above classification reports, all model architectures have lower precision and recall on the `yes` class. __When building the final application, we might want to base in on predicting the `no` class instead
- %timeit magic was used to time the gridsearch and model fit. This will vary quite a bit based on the number of variables we searched over and the associated complexity. Even the data split could make a difference. So it's not an apples to apples comparison and cannot really be used directly for choosing one model architecture vs. another. If we really want to compare the model architectures based on training time, we will have to use a slightly different technique. 

That said, let's summarize what we found:

- All models have an ROC_AUC that is better than baseline. So every one of them will give us some benefit over a blind approach of calling everyone 
- Logistic Regression had the shortest training time
- SVM, when trained with a small data set trained fast while retaining a high ROC_AUC score
- Decision Trees while they take a bit of time to train, shows the best ROC_AUC score. This is with a grid search landing us at a tree depth of 9. With an artificially smaller depth, we could probably reduce training time while retaining ROC_AUC score
- KNN performs the worst in this case and probably should not be used
- Logistic Regression might be the most viable option. It has great traning time, data and/or hyperparameters don't need to be curtailed in any way and all objective scores (ROC_AUC, precision, recall, f1) are pretty good. We only played around with `penalty` though and it would be good to experiment with other hyperparameter

## Next Steps

- Evaluate what the application layer is going to look if we are predicting the `no` class instead of the `yes` class
- Evaluate Decision tree depth further to see if we can maintain ROC_AUC scores while reducing depth, complexity and training time
- Hyperparameter tuning for Logistic Regression so we can improve ROC_AUC score without incurring a significant training time penalty
- Once model architecture is finalized, train and deploy the model to users (sales team, in this case) so they can get predictions on who they should be reaching out to, to sell `fixed term deposits`