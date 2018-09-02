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