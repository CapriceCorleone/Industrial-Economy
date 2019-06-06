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


# ITINERARY_CARRIER_PASSENGERS 每一个航线下每一个航运公司的载客量
# ITINERARY_CARRIER_PASSENGERS% 每一个航线下每一个航运公司的载客量占这一航线总载客量的百分比, MS
# INCUMBENT 在为企业（大于零）
# ITINERARY_CARRIER_REVENUE% MD



path = './industrial economy/'


year = '2018'
quarter = '01'
file = ['COUPON', 'MARKET', 'TICKET']


def get_path(year, quarter, file_num, attr, select=True):
    if select:
        return path + 'selected/' + year + '_' + quarter + '/' + file[file_num] + '_' + year + '_' + quarter + '_selected' + attr
    else:
        return path + 'nonselected/' + year + '_' + quarter + '/' + file[file_num] + '_' + year + '_' + quarter + attr


t_18_01 = pd.read_csv(get_path('2018', '01', 2, '.csv', select=False))
m_18_01 = pd.read_csv(get_path('2018', '01', 1, '.csv', select=False))
c_18_01 = pd.read_csv(get_path('2018', '01', 0, '.csv', select=False))


# selection

m_18_01 = pd.read_csv(get_path('2018', '01', 1, '.csv', select=False))
m_18_01 = m_18_01.drop_duplicates(['ITIN_ID'])

m_18_01['temp1'] = m_18_01['ORIGIN_CITY_MARKET_ID'].apply(lambda x: str(x))
m_18_01['temp2'] = m_18_01['DEST_CITY_MARKET_ID'].apply(lambda x: str(x))
m_18_01['ITINERARY'] = m_18_01['temp1'] + '-' + m_18_01['temp2']
del m_18_01['temp1']
del m_18_01['temp2']






m_18_01['temp1'] = m_18_01['ORIGIN_CITY_MARKET_ID'].apply(lambda x: x)
m_18_01['temp2'] = m_18_01['DEST_CITY_MARKET_ID'].apply(lambda x: x)
df = m_18_01[['temp1', 'temp2']]
df['temp3'] = df['temp1'] - df['temp2']
df['temp4'] = df['temp3'].apply(lambda x: np.sign(x + 0.001))
df['temp5'] = df['temp1'] * df['temp4']
df['temp6'] = df['temp2'] * df['temp4']
df['temp7'] = df['temp5'].apply(lambda x: int(bool(x < 0)))
df['temp8'] = df['temp6'].apply(lambda x: int(bool(x > 0)))
df['temp9'] = df['temp7'] * df['temp1']
df['temp10'] = df['temp8'] * df['temp2']
df['temp11'] = df['temp8'] * df['temp1']
df['temp12'] = df['temp7'] * df['temp2']
df['temp13'] = df['temp9'] + df['temp10']
df['temp14'] = df['temp11'] + df['temp12']
df['temp15'] = df['temp13'].apply(lambda x: str(x))
df['temp16'] = df['temp14'].apply(lambda x: str(x))
df['temp17'] = df['temp15'] + '-' + df['temp16']
m_18_01['ITINERARY_NONDIRECTION'] = df['temp17']
m_18_01['DIRECTED'] = df['temp7']
del m_18_01['temp1']
del m_18_01['temp2']




m_18_01['temp1'] = 1
grouped = m_18_01['temp1'].groupby(m_18_01['ITINERARY_NONDIRECTION']).count()
m_18_01['FLIGHT_COUNTS'] = m_18_01['ITINERARY_NONDIRECTION'].apply(lambda x: grouped[x])
del m_18_01['temp1']



m_18_01_selected = m_18_01[m_18_01['MARKET_DISTANCE'] >= 100]
grouped_passenger = m_18_01_selected['PASSENGERS'].groupby(m_18_01_selected['ITINERARY_NONDIRECTION']).sum()
m_18_01_selected['TOTAL_PASSENGERS'] = m_18_01_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_passenger[x])
m_18_01_selected = m_18_01_selected[m_18_01_selected['TOTAL_PASSENGERS'] >= 900]
m_18_01_selected = m_18_01_selected[['ITIN_ID', 'MKT_ID', 'MARKET_COUPONS',
                                     'YEAR', 'QUARTER', 'ORIGIN_CITY_MARKET_ID',
                                     'DEST_CITY_MARKET_ID', 'PASSENGERS', 'ITINERARY',
                                     'ITINERARY_NONDIRECTION', 'DIRECTED',
                                     'FLIGHT_COUNTS', 'TOTAL_PASSENGERS',
                                     'REPORTING_CARRIER', 'MARKET_FARE', 'MARKET_MILES_FLOWN']]

m_18_01_selected['temp1'] = m_18_01_selected['ITINERARY_NONDIRECTION'] + '~' + m_18_01_selected['REPORTING_CARRIER']
grouped_itinerary_carrier = m_18_01_selected.groupby(['temp1'])['PASSENGERS'].sum()
m_18_01_selected['ITINERARY_CARRIER_PASSENGERS'] = m_18_01_selected['temp1'].apply(lambda x: grouped_itinerary_carrier[x])
m_18_01_selected['ITINERARY_CARRIER_PASSENGERS%'] = m_18_01_selected['ITINERARY_CARRIER_PASSENGERS'] / m_18_01_selected['TOTAL_PASSENGERS']
m_18_01_selected['ITINERARY_CARRIER'] = m_18_01_selected['temp1']
m_18_01_selected['INCUMBENT_INDICATOR_1'] = m_18_01_selected['ITINERARY_CARRIER_PASSENGERS'].apply(lambda x: int(bool(x >= 900)))
m_18_01_selected['INCUMBENT_INDICATOR_2'] = m_18_01_selected['ITINERARY_CARRIER_PASSENGERS%'].apply(lambda x: int(bool(x >= 0.05)))

grouped_incumbent_indicator = m_18_01_selected.groupby(['ITINERARY_NONDIRECTION'])[['INCUMBENT_INDICATOR_1', 'INCUMBENT_INDICATOR_2']].sum()
grouped_incumbent_indicator['INCUMBENT_INDICATOR'] = grouped_incumbent_indicator['INCUMBENT_INDICATOR_1'] + grouped_incumbent_indicator['INCUMBENT_INDICATOR_2']
m_18_01_selected['INCUMBENT'] = m_18_01_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_incumbent_indicator.loc[x, 'INCUMBENT_INDICATOR'])
m_18_01_selected = m_18_01_selected[m_18_01_selected['INCUMBENT'] > 0]

m_18_01_selected.to_csv(path + 'selected/2018_01/MARKET_2018_01_selected.csv')

m_18_01_selected = pd.read_csv(get_path('2018', '01', 1, '.csv'))












t_18_01['YIELD'] = t_18_01['ITIN_YIELD'] * 10 / t_18_01['COUPONS']
df = pd.merge(m_18_01_selected, t_18_01[['ITIN_ID', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'YIELD']], how='left', on=['ITIN_ID'])
df.to_csv(path + 'selected/2018_01/MARKET_2018_01_selected.csv')
m_18_01_selected = pd.read_csv(get_path('2018', '01', 1, '.csv'))

m_18_01_selected['ITINERARY_CARRIER_REVENUE'] = m_18_01_selected['MARKET_FARE'] * m_18_01_selected['PASSENGERS']
grouped_total_revenue = m_18_01_selected.groupby(['REPORTING_CARRIER'])['ITINERARY_CARRIER_REVENUE'].sum()
m_18_01_selected['TOTAL_CARRIER_REVENUE'] = m_18_01_selected['REPORTING_CARRIER'].apply(lambda x: grouped_total_revenue[x])

grouped_total_itinerary_revenue = m_18_01_selected.groupby(['ITINERARY_NONDIRECTION'])['ITINERARY_CARRIER_REVENUE'].sum()
m_18_01_selected['TOTAL_ITINERARY_REVENUE'] = m_18_01_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_total_itinerary_revenue[x])

grouped_itinerary_carrier_revenue = m_18_01_selected.groupby(['ITINERARY_CARRIER'])['ITINERARY_CARRIER_REVENUE'].sum()
m_18_01_selected['TOTAL_ITINERARY_CARRIER_REVENUE'] = m_18_01_selected['ITINERARY_CARRIER'].apply(lambda x: grouped_itinerary_carrier_revenue[x])

m_18_01_selected['ITINERARY_CARRIER_REVENUE%'] = m_18_01_selected['TOTAL_ITINERARY_CARRIER_REVENUE'] / m_18_01_selected['TOTAL_ITINERARY_REVENUE']

m_18_01_selected['temp1'] = 1
grouped_carrier_total_flight = m_18_01_selected.groupby(['REPORTING_CARRIER'])['temp1'].sum()
m_18_01_selected['CARRIER_TOTAL_FLIGHTS'] = m_18_01_selected['REPORTING_CARRIER'].apply(lambda x: grouped_carrier_total_flight[x])

grouped_origin_flight = m_18_01_selected.groupby(['ORIGIN_CITY_MARKET_ID'])['temp1'].sum()
grouped_dest_flight = m_18_01_selected.groupby(['DEST_CITY_MARKET_ID'])['temp1'].sum()
m_18_01_selected['ORIGIN_TOTAL_FLIGHTS'] = m_18_01_selected['ORIGIN_CITY_MARKET_ID'].apply(lambda x: grouped_origin_flight[x] + grouped_dest_flight[x])
m_18_01_selected['DEST_TOTAL_FLIGHTS'] = m_18_01_selected['DEST_CITY_MARKET_ID'].apply(lambda x: grouped_dest_flight[x] + grouped_origin_flight[x])
m_18_01_selected['ORIGIN_DEST_TOTAL_FLIGHTS'] = m_18_01_selected['ORIGIN_TOTAL_FLIGHTS'] + m_18_01_selected['DEST_TOTAL_FLIGHTS']

m_18_01_selected['temp2'] = m_18_01_selected['ORIGIN_CITY_MARKET_ID'].apply(lambda x: str(x))
m_18_01_selected['temp3'] = m_18_01_selected['DEST_CITY_MARKET_ID'].apply(lambda x: str(x))
m_18_01_selected['ORIGIN_CARRIER'] = m_18_01_selected['temp2'] + m_18_01_selected['REPORTING_CARRIER']
m_18_01_selected['DEST_CARRIER'] = m_18_01_selected['temp3'] + m_18_01_selected['REPORTING_CARRIER']
grouped_carrier_origin_flight = m_18_01_selected.groupby(['ORIGIN_CARRIER'])['temp1'].sum()
grouped_carrier_dest_flight = m_18_01_selected.groupby(['DEST_CARRIER'])['temp1'].sum()
m_18_01_selected['ORIGIN_CARRIER_TOTAL_FLIGHTS'] = m_18_01_selected['ORIGIN_CARRIER'].apply(lambda x: grouped_carrier_origin_flight[x])
m_18_01_selected['DEST_CARRIER_TOTAL_FLIGHTS'] = m_18_01_selected['DEST_CARRIER'].apply(lambda x: grouped_carrier_dest_flight[x])
m_18_01_selected['ORIGIN_DEST_CARRIER_TOTAL_FLIGHTS'] = m_18_01_selected['ORIGIN_CARRIER_TOTAL_FLIGHTS'] + m_18_01_selected['DEST_CARRIER_TOTAL_FLIGHTS']

m_18_01_selected['RESOURCE_CENTRALITY'] = m_18_01_selected['ORIGIN_DEST_CARRIER_TOTAL_FLIGHTS'] / m_18_01_selected['ORIGIN_DEST_TOTAL_FLIGHTS']

m_18_01_selected.to_csv(path + 'selected/2018_01/MARKET_2018_01_selected.csv')






m_18_01_selected['MS'] = m_18_01_selected['ITINERARY_CARRIER_PASSENGERS%']
m_18_01_selected['MD'] = m_18_01_selected['ITINERARY_CARRIER_REVENUE%']
m_18_01_selected['RC'] = m_18_01_selected['RESOURCE_CENTRALITY']


m_18_01_selected['temp2'] = m_18_01_selected['ITINERARY_CARRIER_PASSENGERS'].apply(lambda x: int(bool(x >= 900)))
m_18_01_selected['temp3'] = m_18_01_selected['MS'].apply(lambda x: int(bool(x >= 0.05)))
m_18_01_selected['temp4'] = m_18_01_selected['temp2'] + m_18_01_selected['temp3']
m_18_01_selected['INCUMBENT_imt'] = m_18_01_selected['temp4'].apply(lambda x: int(bool(x > 0)))




grouped_rmc = m_18_01_selected.groupby(['ITINERARY_NONDIRECTION'])[['MS', 'MD', 'RC']].max()
m_18_01_selected['MS_max'] = m_18_01_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_rmc.loc[x, 'MS'])
m_18_01_selected['temp2'] = m_18_01_selected['MS'] - m_18_01_selected['MS_max']
m_18_01_selected['LEADER_MS'] = m_18_01_selected['temp2'].apply(lambda x: int(bool(x == 0)))

m_18_01_selected['MD_max'] = m_18_01_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_rmc.loc[x, 'MD'])
m_18_01_selected['temp2'] = m_18_01_selected['MD'] - m_18_01_selected['MD_max']
m_18_01_selected['LEADER_MD'] = m_18_01_selected['temp2'].apply(lambda x: int(bool(x == 0)))

m_18_01_selected['RC_max'] = m_18_01_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_rmc.loc[x, 'RC'])
m_18_01_selected['temp2'] = m_18_01_selected['RC'] - m_18_01_selected['RC_max']
m_18_01_selected['LEADER_RC'] = m_18_01_selected['temp2'].apply(lambda x: int(bool(x == 0)))

m_18_01_selected['CHALLENGER_MS'] = 1 - m_18_01_selected['LEADER_MS']
m_18_01_selected['CHALLENGER_MD'] = 1 - m_18_01_selected['LEADER_MD']
m_18_01_selected['CHALLENGER_RC'] = 1 - m_18_01_selected['LEADER_RC']



class Dummy(object):

    def __init__(self, data, firm, market, time, D):
        self.data = data
        self.firm = firm
        self.market = market
        self.time = time
        self.D = D

    def challenger(self):
        D = self.D
        data = self.data
        firm = self.firm
        market = self.market
        time = self.time
        market_firm = market + '~' + firm
        try:
            result = list(set(list(data[data['ITINERARY_CARRIER'] == market_firm]['CHALLENGER_' + D])))
            if len(result) > 0:
                return result[0]
            else:
                return 0
        except:
            return 0

    def leader(self):
        D = self.D
        data = self.data
        firm = self.firm
        market = self.market
        time = self.time
        market_firm = market + '~' + firm
        try:
            result = list(set(list(data[data['ITINERARY_CARRIER'] == market_firm]['LEADER_' + D])))
            if len(result) > 0:
                return result[0]
            else:
                return 0
        except:
            return 0

    def incumbent(self):
        data = self.data
        firm = self.firm
        market = self.market
        time = self.time
        market_firm = market + '~' + firm
        result = list(set(list(data[data['ITINERARY_CARRIER'] == market_firm]['INCUMBENT_imt'])))
        try:
            if len(result) > 0:
                return result[0]
            else:
                return 0
        except:
            return 0



time = '2018_01'
D = 'MS'

firms = list(set(list(m_18_01_selected['REPORTING_CARRIER'])))
markets = list(set(list(m_18_01_selected['ITINERARY_NONDIRECTION'])))
firms.sort()
markets.sort()

df = m_18_01_selected.drop_duplicates('ITINERARY_CARRIER').groupby('ITINERARY_CARRIER').sum()[['LEADER_' + D, 'CHALLENGER_' + D, 'INCUMBENT_imt']]


def gen_fm_matrix(data, markets, firms, D, res_name):
    df = m_18_01_selected.drop_duplicates('ITINERARY_CARRIER').groupby('ITINERARY_CARRIER').sum()[['LEADER_' + D, 'CHALLENGER_' + D, 'INCUMBENT_imt']]
    result = []
    for market in markets:
        temp = []
        for firm in firms:
            try:
                if res_name == 'INCUMBENT_imt':
                    temp.append(df.loc[market + '~' + firm, 'INCUMBENT_imt'])
                else:
                    temp.append(df.loc[market + '~' + firm, res_name + '_' + D])
            except:
                temp.append(0)
        result.append(temp)
    return result




def gen_mmc(markets, firms, D):

    leader = np.array(gen_fm_matrix(m_18_01_selected, markets, firms, D, 'LEADER'))
    challenger = np.array(gen_fm_matrix(m_18_01_selected, markets, firms, D, 'CHALLENGER'))
    incumbent = np.array(gen_fm_matrix(m_18_01_selected, markets, firms, D, 'INCUMBENT_imt'))

    challenger_reciprocal, challenger_nonreciprocal = [], []
    for firm in firms:
        firm_index = firms.index(firm)
        reciprocal_arr = np.dot(np.dot(challenger, leader.T).T, leader[:, firm_index]) * challenger[:, firm_index]
        nonreciprocal_arr = np.dot(np.dot(incumbent, leader.T).T, challenger[:, firm_index]) * challenger[:, firm_index]
        challenger_reciprocal.append(reciprocal_arr)
        challenger_nonreciprocal.append(nonreciprocal_arr)
    challenger_reciprocal = np.array(challenger_reciprocal).T
    challenger_nonreciprocal = np.array(challenger_nonreciprocal).T


    leader_reciprocal, leader_nonreciprocal = [], []
    for market in markets:
        market_index = markets.index(market)
        challenger_market = np.zeros((len(firms), len(firms)))
        for i in range(len(challenger[market_index, :])):
            challenger_market[i, i] = challenger[market_index, i]
        reciprocal_arr = np.dot(incumbent.T, np.sum(np.dot(leader, challenger_market), axis=1)) * leader[market_index, :] / np.max([1, np.sum(challenger[market_index, :])])
        nonreciprocal_arr = np.dot(incumbent.T, np.sum(np.dot(challenger, challenger_market), axis=1)) * leader[market_index, :] / np.max([1, np.sum(challenger[market_index, :])])
        leader_reciprocal.append(reciprocal_arr)
        leader_nonreciprocal.append(nonreciprocal_arr)
    leader_reciprocal = np.array(leader_reciprocal)
    leader_nonreciprocal = np.array(leader_nonreciprocal)

    return challenger_reciprocal, challenger_nonreciprocal, leader_reciprocal, leader_nonreciprocal



def fill_mmc(data, markets, firms, D):
    challenger_reciprocal, challenger_nonreciprocal, leader_reciprocal, leader_nonreciprocal = gen_mmc(markets, firms, D)
    data['CHA_RECI_' + D] = data['ITINERARY_CARRIER'].apply(lambda x: challenger_reciprocal[markets.index(x.split('~')[0]), firms.index(x.split('~')[1])])
    data['CHA_NONRECI_' + D] = data['ITINERARY_CARRIER'].apply(lambda x: challenger_nonreciprocal[markets.index(x.split('~')[0]), firms.index(x.split('~')[1])])
    data['LEA_RECI_' + D] = data['ITINERARY_CARRIER'].apply(lambda x: leader_reciprocal[markets.index(x.split('~')[0]), firms.index(x.split('~')[1])])
    data['LEA_NONRECI_' + D] = data['ITINERARY_CARRIER'].apply(lambda x: leader_nonreciprocal[markets.index(x.split('~')[0]), firms.index(x.split('~')[1])])
    return data


m_18_01_selected = fill_mmc(m_18_01_selected, markets, firms, 'MS')
m_18_01_selected = fill_mmc(m_18_01_selected, markets, firms, 'MD')
m_18_01_selected = fill_mmc(m_18_01_selected, markets, firms, 'RC')

m_18_01_selected.to_csv(path + 'selected/2018_01/MARKET_2018_01_selected.csv')


m_18_01_selected['temp1'] = 1
grouped_itin_carrier_flight = m_18_01_selected.groupby(['ITINERARY_CARRIER'])['temp1'].sum()
m_18_01_selected['FREQUENCY'] = m_18_01_selected['ITINERARY_CARRIER'].apply(lambda x: grouped_itin_carrier_flight.loc[x])



m_18_01_selected = pd.merge(m_18_01_selected, c_18_01[['ITIN_ID', 'FARE_CLASS']], how='left', on=['ITIN_ID'])

grouped_market_size = m_18_01_selected.groupby(['ITINERARY_NONDIRECTION'])['PASSENGERS'].sum()
m_18_01_selected['MARKET_SIZE'] = m_18_01_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_market_size.loc[x])

grouped_firm_size = m_18_01_selected.groupby(['REPORTING_CARRIER'])['PASSENGERS'].sum()
m_18_01_selected['temp2'] = m_18_01_selected['REPORTING_CARRIER'].apply(lambda x: grouped_firm_size.loc[x])
m_18_01_selected['FIRM_SIZE'] = m_18_01_selected['CARRIER_TOTAL_FLIGHTS'] - m_18_01_selected['temp2']

m_18_01_selected['temp1'] = 1
grouped_carrier_origin_passengers = m_18_01_selected.groupby(['ORIGIN_CARRIER'])['temp1'].sum()
grouped_carrier_dest_passengers = m_18_01_selected.groupby(['DEST_CARRIER'])['temp1'].sum()
m_18_01_selected['ORIGIN_CARRIER_TOTAL_PASSENGERS'] = m_18_01_selected['ORIGIN_CARRIER'].apply(lambda x: grouped_carrier_origin_passengers[x])
m_18_01_selected['DEST_CARRIER_TOTAL_PASSENGERS'] = m_18_01_selected['DEST_CARRIER'].apply(lambda x: grouped_carrier_dest_passengers[x])
m_18_01_selected['ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS'] = m_18_01_selected['ORIGIN_CARRIER_TOTAL_FLIGHTS'] + m_18_01_selected['DEST_CARRIER_TOTAL_FLIGHTS']

incumbent = np.array(gen_fm_matrix(m_18_01_selected, markets, firms, D, 'INCUMBENT_imt'))
grouped_firm_MS = m_18_01_selected.drop_duplicates('REPORTING_CARRIER').groupby(['REPORTING_CARRIER'])['MS'].sum()
grouped_itin_carrier_MS = m_18_01_selected.drop_duplicates('ITINERARY_CARRIER').groupby(['ITINERARY_CARRIER'])['MS'].sum()
m_18_01_selected['AMS_OUTSIDE'] = m_18_01_selected['ITINERARY_CARRIER'].apply(lambda x: (grouped_firm_MS.loc[x.split('~')[1]] - grouped_itin_carrier_MS.loc[x]) / np.sum(incumbent[:, firms.index(x.split('~')[1])]))


















# plotting

data_m = pd.read_csv(get_path('2017', '01', 1, '.csv'))
df = data_m.drop_duplicates('ITINERARY_CARRIER')[['ITINERARY_CARRIER', 'REPORTING_CARRIER', 'MS']]
grouped_itin_carrier_MS = df.groupby(['ITINERARY_CARRIER'])['MS'].sum()
grouped_firm_MS = df.groupby(['REPORTING_CARRIER'])['MS'].sum()

ms_firm, ms_firm_market = [], []
for firm in firms:
    ms_firm.append(grouped_firm_MS.loc[firm])
    temp = []
    for market in markets:
        try:
            temp.append(grouped_itin_carrier_MS.loc[market + '~' + firm])
        except:
            temp.append(0)
    ms_firm_market.append(temp)

ms_firm, ms_firm_market = np.array(ms_firm), np.array(ms_firm_market)



plt.figure(figsize=(9, 9))
x, y = [], []
scale_firm = ms_firm.mean()
scale_firm_max = np.max(ms_firm / scale_firm)
top = scale_firm_max * 10
x.append(np.random.rand() * top)
y.append(np.random.rand() * top)
for firm in firms:
    firm_index = firms.index(firm)
    radius = ms_firm[firm_index] / scale_firm
    plt.scatter(x[firm_index], y[firm_index], color='black', s=radius)
    while True:
        new_x, new_y = np.random.rand() * top, np.random.rand() * top
        temp_x, temp_y = np.array(x), np.array(y)
        distance_min = np.sqrt(np.min((temp_x - new_x) ** 2 + (temp_y - new_y) ** 2))
        if distance_min > 1.5 * radius:
            x.append(new_x)
            y.append(new_y)
            break








df = pd.read_csv(get_path('2014', '01', 1, '.csv', select=True))
for year in range(2015, 2019):
    temp = pd.read_csv(get_path(str(year), '01', 1, '.csv', select=True))
    pd.concat([df, temp], ignore_index=True)

df.dropna(axis=0, how='any')
























