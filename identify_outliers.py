import pickle
import sys
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from operator import itemgetter

### Read in data dictionary, convert to numpy array
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

# Before removing outliers
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.title("Before removing outlier")
matplotlib.pyplot.show()

"""
We can see from the plot given from the above code that there is one big outlier
in the data with salary > 25,000,000 and bonus > 10,000,000.

Checking the the PDF document (enron61702insiderpay.pdf) for the key of this data 
point we see that the key is "TOTAL" which is the total of all the values in the 
dataset. It is a spreadsheet error and needs to be cleaned.
"""

# Removing outlier
data_dict.pop('TOTAL',0)

# After removing outlier
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.title("After removing outlier")
matplotlib.pyplot.show()

"""
Again after cleaning the first outlier, we can see there are around 7 points that 
look like outliers.

Taking a look at the names for these data points.
"""

for key,value in data_dict.items():
    if float(value['salary'])>600000. or float(value['bonus'])>4000000.:
        print "Name: ",key
        print "Salary: ",value['salary']
        print "Bonus: ",value['bonus']
        print "-----"

"""
As we can clearly see from the names, none of these data points are outliers that 
need to be cleaned from the dataset.
"""
