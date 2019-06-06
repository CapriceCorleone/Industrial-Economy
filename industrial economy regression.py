#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# working path cd /Users/CapriceCorleone/PycharmProjects/SummerVacation


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import NullFormatter
import datetime as dt
import json, logging
from geopy.geocoders import Nominatim

path = './industrial economy/'

data = pd.read_csv(path + '_compiled.csv')

data_se = data.dropna()
data_se = data.drop_duplicates(['YEAR', 'ITINERARY_CARRIER'])


"""
Unnamed: 0, Unnamed: 0.1, ITIN_ID, MARKET_COUPONS, YEAR, QUARTER, PASSENGERS, ITINERARY_NONDIRECTION, REPORTING_CARRIER, 
MARKET_FARE, MARKET_MILES_FLOWN, temp1, ITINERARY_CARRIER_PASSENGERS, ITINERARY_CARRIER, ROUNDTRIP, ITIN_YIELD, ITIN_FARE, 
YIELD, temp2, temp3, MS, MD, RC, temp4, INCUMBENT_imt, LEADER_MS, LEADER_MD, LEADER_RC, CHALLENGER_MS, CHALLENGER_MD, 
CHALLENGER_RC, FREQUENCY, FARE_CLASS, MARKET_SIZE, FIRM_SIZE, ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS, CHA_RECI_MS, CHA_NONRECI_MS, 
LEA_RECI_MS, LEA_NONRECI_MS, CHA_RECI_MD, CHA_NONRECI_MD, LEA_RECI_MD, LEA_NONRECI_MD, CHA_RECI_RC, CHA_NONRECI_RC, LEA_RECI_RC, 
LEA_NONRECI_RC, AMS_OUTSIDE
"""

data_se = data_se[['YEAR', 'ITINERARY_CARRIER', 'YIELD', 'MS', 'FREQUENCY', 'ROUNDTRIP', 'FARE_CLASS', 'MARKET_SIZE', 'FIRM_SIZE',
                   'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE', 'CHA_RECI_MS', 'CHA_NONRECI_MS', 'LEA_RECI_MS',
                   'LEA_NONRECI_MS', 'CHA_RECI_MD', 'CHA_NONRECI_MD', 'LEA_RECI_MD', 'LEA_NONRECI_MD', 'CHA_RECI_RC',
                   'CHA_NONRECI_RC', 'LEA_RECI_RC', 'LEA_NONRECI_RC']]
data_se = data_se.dropna()

data_se.to_csv(path + '_compiled_se.csv')


def get_dummy(data, column):

    category_key = list(set(list(data[column])))
    category_key.sort()
    category_value = [i for i in range(len(category_key))]
    category_dict = dict(zip(category_key, category_value))

    data[column + '_dummy'] = data[column].apply(lambda x: category_dict[x])

    return data


category_columns = ['ROUNDTRIP', 'FARE_CLASS']
for column in category_columns:
    data_se = get_dummy(data_se, column)


D = ['MS', 'MD', 'RC']


def get_total_MMC(data, d):

    data['CHA_TOTAL_' + d] = data['CHA_RECI_' + d] + data['CHA_NONRECI_' + d]
    data['LEA_TOTAL_' + d] = data['LEA_RECI_' + d] + data['LEA_NONRECI_' + d]

    return data


for d in D:
    data_se = get_total_MMC(data_se, d)


model_MS_1 = sm.OLS(data_se['YIELD'], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                               'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE',
                                                               'CHA_TOTAL_MS']])).fit()
model_MS_2 = sm.OLS(data_se['YIELD'], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                               'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE',
                                                               'CHA_RECI_MS', 'CHA_NONRECI_MS']])).fit()
model_MS_1.summary()



def get_model_sum(data_se, y, d, identity, num):
    """
    :param y: YIELD, MS
    :param d: MS, MD, RC
    :param num: 0, 1, 2
    :param identity: CHA, LEA
    :return: data_se
    """
    if y == 'YIELD':
        if identity == 'CHA':
            if num == 0:
                model = sm.OLS(data_se[y], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                                    'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE'
                                                                    ]])).fit()
            elif num == 1:
                model = sm.OLS(data_se[y], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                                    'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE',
                                                                    'CHA_TOTAL_' + d]])).fit()
            elif num == 2:
                model = sm.OLS(data_se[y], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                                    'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE',
                                                                    'CHA_RECI_' + d, 'CHA_NONRECI_' + d]])).fit()
            else:
                print("NUM ERROR: num out of bounds")

        elif identity == 'LEA':
            if num == 0:
                model = sm.OLS(data_se[y], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                                    'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE'
                                                                    ]])).fit()
            elif num == 1:
                model = sm.OLS(data_se[y], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                                    'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE',
                                                                    'LEA_TOTAL_' + d]])).fit()
            elif num == 2:
                model = sm.OLS(data_se[y], sm.add_constant(data_se[['FREQUENCY', 'ROUNDTRIP_dummy', 'FARE_CLASS_dummy', 'MARKET_SIZE',
                                                                    'FIRM_SIZE', 'ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS', 'AMS_OUTSIDE',
                                                                    'LEA_RECI_' + d, 'LEA_NONRECI_' + d]])).fit()
            else:
                print("NUM ERROR: num out of bounds")

    else:
        print("Y ERROR: y out of bounds")

    print(y, d, identity, num)

    return model


get_model_sum(data_se, 'YIELD', 'MS', 'CHA', 1).summary()