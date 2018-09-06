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