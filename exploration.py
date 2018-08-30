import pickle
import math
import pprint

enron_data = pickle.load(open("final_project_dataset.pkl", "r"))

features_used = ['bonus',
                'deferral_payments',
                'deferred_income',
                'director_fees',
                'exercised_stock_options',
                'expenses',
                'from_messages',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'loan_advances',
                'long_term_incentive',
                'other',
                'percent_from_poi_to_this_person',
                'percent_from_this_person_to_poi',
                'restricted_stock',
                'restricted_stock_deferred',
                'salary',
                'shared_receipt_with_poi',
                'to_messages',
                'total_payments',
                'total_stock_value']

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

for key,value in data_dict.items():

    from_poi_to_this_person_isnan = math.isnan(float(value['from_poi_to_this_person']))
    from_this_person_to_poi_isnan = math.isnan(float(value['from_this_person_to_poi']))

    if((not from_poi_to_this_person_isnan) and (not from_this_person_to_poi_isnan)):
        percent_from_poi_to_this_person = value['from_poi_to_this_person']/value['from_messages']
        percent_from_this_person_to_poi = value['from_this_person_to_poi']/value['to_messages']

        value['percent_from_poi_to_this_person'] = percent_from_poi_to_this_person
        value['percent_from_this_person_to_poi'] = percent_from_this_person_to_poi

    else:
        value['percent_from_poi_to_this_person'] = 'NaN'
        value['percent_from_this_person_to_poi'] = 'NaN'

print "Number of data points: ",len(enron_data)

nb_of_poi = 0
nb_of_non_poi = 0

for key,value in data_dict.items():
    if value['poi']:
        nb_of_poi += 1
    else:
        nb_of_non_poi += 1

print "Number of POI: ",nb_of_poi
print "Number of non-POI: ",nb_of_non_poi

print "Number of features used: ",len(features_used)

nan_values = {}
nan_values = dict((el,0) for el in features_used)

for feature in features_used:
    for key,value in data_dict.items():
        if math.isnan(float(value[feature])):
            nan_values[feature] += 1

print "\nNumber of NaN values:"
pprint.pprint(nan_values)
            
        