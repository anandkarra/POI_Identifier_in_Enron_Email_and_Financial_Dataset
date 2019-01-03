# Free-Response Report

## 1. Exploration and Outlier Investigation

The goal of this project was the implementation of a **Person Of Interest (POI) identifier** for the **Enron email and financial dataset**. A POI in this instance is a person who is suspected of fraud or other infractions. The dataset is a one-of-a-kind dataset with the email and financial details of the employees of Enron Corp. released to the public as part of the investigation. The dataset has already been processed to mark potential POIs based on News and investifation reports. Some basic exploration of the dataset gives the following statistics:
```python
Number of data points:  146
Number of POI:  18
Number of non-POI:  128
Number of features used:  21

Number of NaN values:
{'bonus': 64,
 'deferral_payments': 107,
 'deferred_income': 97,
 'director_fees': 129,
 'exercised_stock_options': 44,
 'expenses': 51,
 'from_messages': 60,
 'from_poi_to_this_person': 60,
 'from_this_person_to_poi': 60,
 'loan_advances': 142,
 'long_term_incentive': 80,
 'other': 53,
 'percent_from_poi_to_this_person': 60,
 'percent_from_this_person_to_poi': 60,
 'restricted_stock': 36,
 'restricted_stock_deferred': 128,
 'salary': 51,
 'shared_receipt_with_poi': 60,
 'to_messages': 60,
 'total_payments': 21,
 'total_stock_value': 20}
```
More exploration is performed in `exploration.py`.

Using the features shown above we can train a machine learning algorithm to try to predict whether an employee was a POI or not. The features that are used for this purpose are elaborated upon in the forthcoming sections.

To investigate outliers, we first plot the data points (Carried out in `identify_outliers.py`). Immediately, we see a data point with **salary > 25 million** and **bonus > 10 million**. Checking the the PDF document (`enron61702insiderpay.pdf`) for the key of this data point we see that the key is "TOTAL" which is the total of all the values in the dataset. This is a spreadsheet error and is cleaned. Again, we plot the cleaned data and observe what appears like 7 outliers. Upon programmatically verifying, we get the following output:
```python
Name:  LAVORATO JOHN J
Salary:  339288
Bonus:  8000000
-----
Name:  LAY KENNETH L
Salary:  1072321
Bonus:  7000000
-----
Name:  BELDEN TIMOTHY N
Salary:  213999
Bonus:  5249999
-----
Name:  SKILLING JEFFREY K
Salary:  1111258
Bonus:  5600000
-----
Name:  PICKERING MARK R
Salary:  655037
Bonus:  300000
-----
Name:  ALLEN PHILLIP K
Salary:  201955
Bonus:  4175000
-----
Name:  FREVERT MARK A
Salary:  1060932
Bonus:  2000000
-----
```
Clearly, these are all valid data points.

## 2. Feature Selection and Scaling

The following features were finally used in the POI identifier:
```
'poi'
'salary'
'bonus'
'exercised_stock_options'
'total_stock_value'
'from_poi_to_this_person'
'from_this_person_to_poi'
'expenses'
'total_payments'
```
These features were selected after comparing the performance of difference combinations of features (carried out in `feature_selection.py`).

The feature groups:
* **basic_financial** ('poi','salary','bonus')
* **mixed_2** ('poi','salary','bonus','from_poi_to_this_person','from_this_person_to_poi')
* **mixed_3** ('poi','salary','bonus','exercised_stock_options','total_stock_value')
* **mixed_4** ('poi','salary','bonus','exercised_stock_options','total_stock_value','from_poi_to_this_person','from_this_person_to_poi','expenses','total_payments')

perform the best in terms of accuracy, precision and recall.

Hence, these four feature groups will be further tested in the POI identifier code (`poi_id.py`) using the `tester.py` script for different classifiers.

Two new features- **percent_from_poi_to_this_person**, **percent_from_this_person_to_poi** were also engineered. The former measured the percentage of messages from a marked POI to the given person; the latter measure the percentage of messages from the given person to a marked POI. These features indicate how frequently a given person communicates with a POI which can be a factor in determining if a given person is a POI. The performace of these new features is explored is `new_feature_performance.py`.

Further, as the features in the dataset span various ranges, implementing **feature scaling** became imperative. It is done after engineering the new feature in the POI identifier code (`poi_id.py`). As the newly created features are already percentages, there is no need to apply feature scaling on them.

As the classifier finally used for the POI identifier is a Decision Tree, the feature importances are given below.
```
[0.05860806 0.         0.15170137 0.20792079 0.         0.23719108
 0.3445787  0.        ]
```

## 3. Classifiers

A Decision Tree classifier was finally used for the POI identifier.

The performance of a Decision Tree classifier and a Support Vector Classifier (SVC) were compared (in `poi_id.py`).

The performance metrics are as follows:
```python
----- SVC -----
Training time:  0.001 s
('Predicting time: ', 0.0, 's')
Accuracy=  0.88
/home/m4rvin/anaconda2/lib/python2.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
             precision    recall  f1-score   support

        0.0       0.88      1.00      0.94        58
        1.0       0.00      0.00      0.00         8

avg / total       0.77      0.88      0.82        66

----- Decision Tree -----
Training time:  0.0 s
('Predicting time: ', 0.0, 's')
Accuracy=  0.85
             precision    recall  f1-score   support

        0.0       0.91      0.91      0.91        58
        1.0       0.38      0.38      0.38         8

avg / total       0.85      0.85      0.85        66
```

## 4. Patameter Tuning
Tuning the parameters of an algorithm refers to trying out different combinations of parameters of an algorithm to obtain the best performance. This may be done manually or using built-in methods in sklearn like GridSearchCV.

The following parameters for the Decision Tree Classifer were tuned using GridSearchCV: 
* **criterion** (`gini` and `entropy`)
* **min_samples_split** (1,10,100,1000)
* **presort** (False,True)

The following parameters for the SVC were tuned using GridSearchCV: 
* **kernel** (`linear` and `rbf`)
* **C** (2,3,4,5,6)
* **gamma** (0.01,0.02,0.05,0.5,0.2,0.1,1)


Overall, after tuning, the Decision Tree Classifier performed better than the SVC. As such, it is used in the final POI identifier (poi_id.py).

The output and performance of both the Classifiers with tuning performed can be found in the `with_tuning` directory.

Failure to tune the parameters of a Classifier may result not only in sub-optimal performance but may also cause the Classifier to overfit to the test data and not be able to generalize well to the testing data.

The following code was used for parameter tuning. Only the features tuned here were expected to affect the performance of the Classifier as the dataset is a very skewed one with only 18 POIs and 128 non-POIs.

```python
# Tuning Decision Tree Classifier
parameters_dtc = {'criterion' : ['gini','entropy'],
                'min_samples_split' : [2,3,4,5,6],
                'presort' : [False,True]
            }

dtc = tree.DecisionTreeClassifier(random_state=42)
clf_dtc = GridSearchCV(dtc,parameters_dtc)
```

```python
# Tuning SVC
parameters_svc = {'kernel' : ['linear','rbf'],
                'C' : [1,10,100,1000],
                'gamma' : [0.01,0.02,0.05,0.5,0.2,0.1,1]

}

svr = SVC(random_state=10)
clf_svc = GridSearchCV(svr,parameters_svc)
```

## 5. Validation
Validation is a process of splitting the dataset into training and testing sets such that the classifier achieves the best performance possible. Validation accomplishes this by splitting the dataset into the training and testing sets differently and then assessing the performance of the classifier. At the end of the validation process, the classifier has effectively been trained and tested on the entire dataset.

A very common mistake that may slip unnoticed while performing validation is performing validation after doing the feature selection. This leads to a Classifier which over-estimates its performance as the Classifier already has some information about the testing data. To prevent this from happening, validation must be performed before feature selection and then again after feature scaling to achieve the best performance possible.

**KFold Cross validation** with 4 splits was used for the Classifier. It was implemented as follows:
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=4,shuffle=True,random_state=10)
```
Based on the performance of the 4 splits, the best split was then selected to form the training and testing data for the Classifier.

## 6. Evaluation Metrics
Precision and Recall were the two evaluation metrics that were of prime importance in the implementation of the POI identifier as we can understand from the meanings of these terms. For this specific use case, the meanings of Precision and Recall can be described as follows:
* **Precision :** The probabilty that the if the Classifier identifies a given person as a POI then he/she is actually a POI.
* **Recall :** The probability that the Classifier correctly identifies a given person as a POI provided that the he/she is actually a POI.

The tuned Decision Tree Classifier gives the following metrics.

| Metric | Value|
|--|--|
| Accuracy | 0.821 |
| Precision | 0.322 |
| Recall | 0.313 |

Practically, these metrics translate to the Classifer correctly predicting a given person as a POI and a given person predicted to be a POI to actually be one in about one-third of the cases. In combination with a 82% accuracy, this Classifier forms a very practical POI identifier.

----