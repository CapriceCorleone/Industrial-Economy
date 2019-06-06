#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# working path cd /Users/CapriceCorleone/PycharmProjects/SummerVacation


import pandas as pd
import numpy as np



def get_path(year, quarter, file_num, attr, select=True):
    if select:
        return path + 'selected/' + year + '_' + quarter + '/' + file[file_num] + '_' + year + '_' + quarter + '_selected' + attr
    else:
        return path + 'nonselected/' + year + '_' + quarter + '/' + file[file_num] + '_' + year + '_' + quarter + attr



# selection
def selection(time):
    year, quarter = time.split('_')

    data_m = pd.read_csv(get_path(year, quarter, 1, '.csv', select=False))
    data_m = data_m.drop_duplicates(['ITIN_ID'])
    
    data_m['temp1'] = data_m['ORIGIN_CITY_MARKET_ID'].apply(lambda x: str(x))
    data_m['temp2'] = data_m['DEST_CITY_MARKET_ID'].apply(lambda x: str(x))
    data_m['ITINERARY'] = data_m['temp1'] + '-' + data_m['temp2']
    del data_m['temp1']
    del data_m['temp2']
    
    data_m['temp1'] = data_m['ORIGIN_CITY_MARKET_ID'].apply(lambda x: x)
    data_m['temp2'] = data_m['DEST_CITY_MARKET_ID'].apply(lambda x: x)
    df = data_m[['temp1', 'temp2']]
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
    data_m['ITINERARY_NONDIRECTION'] = df['temp17']
    data_m['DIRECTED'] = df['temp7']
    del data_m['temp1']
    del data_m['temp2']
    df = 0
    
    data_m['temp1'] = 1
    grouped = data_m['temp1'].groupby(data_m['ITINERARY_NONDIRECTION']).count()
    data_m['FLIGHT_COUNTS'] = data_m['ITINERARY_NONDIRECTION'].apply(lambda x: grouped[x])
    del data_m['temp1']
    grouped = 0

    data_m_selected = data_m[data_m['MARKET_DISTANCE'] >= 100]
    grouped_passenger = data_m_selected['PASSENGERS'].groupby(data_m_selected['ITINERARY_NONDIRECTION']).sum()
    data_m_selected['TOTAL_PASSENGERS'] = data_m_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_passenger[x])
    data_m_selected = data_m_selected[data_m_selected['TOTAL_PASSENGERS'] >= 900]
    data_m_selected = data_m_selected[['ITIN_ID', 'MARKET_COUPONS', 'YEAR', 'QUARTER', 'ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID',
                                       'PASSENGERS', 'ITINERARY_NONDIRECTION', 'TOTAL_PASSENGERS', 'REPORTING_CARRIER', 'MARKET_FARE',
                                       'MARKET_MILES_FLOWN']]
    grouped_passenger = 0

    data_m_selected['REPORTING_CARRIER'] = data_m_selected['REPORTING_CARRIER'].apply(lambda x: str(x))
    data_m_selected['temp1'] = data_m_selected['ITINERARY_NONDIRECTION'] + '~' + data_m_selected['REPORTING_CARRIER']
    grouped_itinerary_carrier = data_m_selected.groupby(['temp1'])['PASSENGERS'].sum()
    data_m_selected['ITINERARY_CARRIER_PASSENGERS'] = data_m_selected['temp1'].apply(lambda x: grouped_itinerary_carrier[x])
    data_m_selected['ITINERARY_CARRIER_PASSENGERS%'] = data_m_selected['ITINERARY_CARRIER_PASSENGERS'] / data_m_selected['TOTAL_PASSENGERS']
    data_m_selected['ITINERARY_CARRIER'] = data_m_selected['temp1']
    data_m_selected['INCUMBENT_INDICATOR_1'] = data_m_selected['ITINERARY_CARRIER_PASSENGERS'].apply(lambda x: int(bool(x >= 900)))
    data_m_selected['INCUMBENT_INDICATOR_2'] = data_m_selected['ITINERARY_CARRIER_PASSENGERS%'].apply(lambda x: int(bool(x >= 0.05)))
    grouped_itinerary_carrier = 0

    del data_m_selected['TOTAL_PASSENGERS']
    
    grouped_incumbent_indicator = data_m_selected.groupby(['ITINERARY_NONDIRECTION'])[['INCUMBENT_INDICATOR_1', 'INCUMBENT_INDICATOR_2']].sum()
    grouped_incumbent_indicator['INCUMBENT_INDICATOR'] = grouped_incumbent_indicator['INCUMBENT_INDICATOR_1'] + grouped_incumbent_indicator['INCUMBENT_INDICATOR_2']
    data_m_selected['INCUMBENT'] = data_m_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_incumbent_indicator.loc[x, 'INCUMBENT_INDICATOR'])
    data_m_selected = data_m_selected[data_m_selected['INCUMBENT'] > 0]
    grouped_incumbent_indicator = 0

    del data_m_selected['INCUMBENT_INDICATOR_1']
    del data_m_selected['INCUMBENT_INDICATOR_2']
    del data_m_selected['INCUMBENT']

    data_t = pd.read_csv(get_path(year, quarter, 2, '.csv', select=False))

    data_t['YIELD'] = data_t['ITIN_YIELD'] * 10 / data_t['COUPONS']
    data_m_selected = pd.merge(data_m_selected, data_t[['ITIN_ID', 'ROUNDTRIP', 'ITIN_YIELD', 'ITIN_FARE', 'YIELD']], how='left', on=['ITIN_ID'])
    
    data_m_selected['ITINERARY_CARRIER_REVENUE'] = data_m_selected['MARKET_FARE'] * data_m_selected['PASSENGERS']
    
    grouped_total_itinerary_revenue = data_m_selected.groupby(['ITINERARY_NONDIRECTION'])['ITINERARY_CARRIER_REVENUE'].sum()
    data_m_selected['TOTAL_ITINERARY_REVENUE'] = data_m_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_total_itinerary_revenue[x])
    grouped_total_itinerary_revenue = 0
    
    grouped_itinerary_carrier_revenue = data_m_selected.groupby(['ITINERARY_CARRIER'])['ITINERARY_CARRIER_REVENUE'].sum()
    data_m_selected['TOTAL_ITINERARY_CARRIER_REVENUE'] = data_m_selected['ITINERARY_CARRIER'].apply(lambda x: grouped_itinerary_carrier_revenue[x])
    grouped_itinerary_carrier_revenue = 0
    
    data_m_selected['ITINERARY_CARRIER_REVENUE%'] = data_m_selected['TOTAL_ITINERARY_CARRIER_REVENUE'] / data_m_selected['TOTAL_ITINERARY_REVENUE']
    
    data_m_selected['temp1'] = 1
    grouped_carrier_total_flight = data_m_selected.groupby(['REPORTING_CARRIER'])['temp1'].sum()
    data_m_selected['CARRIER_TOTAL_FLIGHTS'] = data_m_selected['REPORTING_CARRIER'].apply(lambda x: grouped_carrier_total_flight[x])
    grouped_carrier_total_flight = 0
    
    grouped_origin_flight = data_m_selected.groupby(['ORIGIN_CITY_MARKET_ID'])['temp1'].sum()
    grouped_dest_flight = data_m_selected.groupby(['DEST_CITY_MARKET_ID'])['temp1'].sum()
    data_m_selected['ORIGIN_TOTAL_FLIGHTS'] = data_m_selected['ORIGIN_CITY_MARKET_ID'].apply(lambda x: grouped_origin_flight[x] + grouped_dest_flight[x])
    data_m_selected['DEST_TOTAL_FLIGHTS'] = data_m_selected['DEST_CITY_MARKET_ID'].apply(lambda x: grouped_dest_flight[x] + grouped_origin_flight[x])
    data_m_selected['ORIGIN_DEST_TOTAL_FLIGHTS'] = data_m_selected['ORIGIN_TOTAL_FLIGHTS'] + data_m_selected['DEST_TOTAL_FLIGHTS']
    grouped_origin_flight, grouped_dest_flight = 0, 0

    data_m_selected['temp2'] = data_m_selected['ORIGIN_CITY_MARKET_ID'].apply(lambda x: str(x))
    data_m_selected['temp3'] = data_m_selected['DEST_CITY_MARKET_ID'].apply(lambda x: str(x))
    data_m_selected['ORIGIN_CARRIER'] = data_m_selected['temp2'] + data_m_selected['REPORTING_CARRIER']
    data_m_selected['DEST_CARRIER'] = data_m_selected['temp3'] + data_m_selected['REPORTING_CARRIER']
    grouped_carrier_origin_flight = data_m_selected.groupby(['ORIGIN_CARRIER'])['temp1'].sum()
    grouped_carrier_dest_flight = data_m_selected.groupby(['DEST_CARRIER'])['temp1'].sum()
    data_m_selected['ORIGIN_CARRIER_TOTAL_FLIGHTS'] = data_m_selected['ORIGIN_CARRIER'].apply(lambda x: grouped_carrier_origin_flight[x])
    data_m_selected['DEST_CARRIER_TOTAL_FLIGHTS'] = data_m_selected['DEST_CARRIER'].apply(lambda x: grouped_carrier_dest_flight[x])
    data_m_selected['ORIGIN_DEST_CARRIER_TOTAL_FLIGHTS'] = data_m_selected['ORIGIN_CARRIER_TOTAL_FLIGHTS'] + data_m_selected['DEST_CARRIER_TOTAL_FLIGHTS']
    grouped_carrier_origin_flight, grouped_carrier_dest_flight = 0, 0

    del data_m_selected['ORIGIN_CITY_MARKET_ID']
    del data_m_selected['DEST_CITY_MARKET_ID']
    del data_m_selected['ITINERARY_CARRIER_REVENUE']
    del data_m_selected['TOTAL_ITINERARY_REVENUE']
    del data_m_selected['TOTAL_ITINERARY_CARRIER_REVENUE']
    del data_m_selected['ORIGIN_TOTAL_FLIGHTS']
    del data_m_selected['DEST_TOTAL_FLIGHTS']

    data_m_selected['RESOURCE_CENTRALITY'] = data_m_selected['ORIGIN_DEST_CARRIER_TOTAL_FLIGHTS'] / data_m_selected['ORIGIN_DEST_TOTAL_FLIGHTS']

    data_m_selected['MS'] = data_m_selected['ITINERARY_CARRIER_PASSENGERS%']
    data_m_selected['MD'] = data_m_selected['ITINERARY_CARRIER_REVENUE%']
    data_m_selected['RC'] = data_m_selected['RESOURCE_CENTRALITY']

    del data_m_selected['ITINERARY_CARRIER_PASSENGERS%']
    del data_m_selected['ITINERARY_CARRIER_REVENUE%']
    del data_m_selected['RESOURCE_CENTRALITY']
    del data_m_selected['ORIGIN_DEST_TOTAL_FLIGHTS']
    del data_m_selected['ORIGIN_DEST_CARRIER_TOTAL_FLIGHTS']
    
    data_m_selected['temp2'] = data_m_selected['ITINERARY_CARRIER_PASSENGERS'].apply(lambda x: int(bool(x >= 900)))
    data_m_selected['temp3'] = data_m_selected['MS'].apply(lambda x: int(bool(x >= 0.05)))
    data_m_selected['temp4'] = data_m_selected['temp2'] + data_m_selected['temp3']
    data_m_selected['INCUMBENT_imt'] = data_m_selected['temp4'].apply(lambda x: int(bool(x > 0)))

    grouped_rmc = data_m_selected.groupby(['ITINERARY_NONDIRECTION'])[['MS', 'MD', 'RC']].max()
    data_m_selected['MS_max'] = data_m_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_rmc.loc[x, 'MS'])
    data_m_selected['temp2'] = data_m_selected['MS'] - data_m_selected['MS_max']
    data_m_selected['LEADER_MS'] = data_m_selected['temp2'].apply(lambda x: int(bool(x == 0)))

    data_m_selected['MD_max'] = data_m_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_rmc.loc[x, 'MD'])
    data_m_selected['temp2'] = data_m_selected['MD'] - data_m_selected['MD_max']
    data_m_selected['LEADER_MD'] = data_m_selected['temp2'].apply(lambda x: int(bool(x == 0)))
    
    data_m_selected['RC_max'] = data_m_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_rmc.loc[x, 'RC'])
    data_m_selected['temp2'] = data_m_selected['RC'] - data_m_selected['RC_max']
    data_m_selected['LEADER_RC'] = data_m_selected['temp2'].apply(lambda x: int(bool(x == 0)))

    del data_m_selected['MS_max']
    del data_m_selected['MD_max']
    del data_m_selected['RC_max']
    
    data_m_selected['CHALLENGER_MS'] = 1 - data_m_selected['LEADER_MS']
    data_m_selected['CHALLENGER_MD'] = 1 - data_m_selected['LEADER_MD']
    data_m_selected['CHALLENGER_RC'] = 1 - data_m_selected['LEADER_RC']
    grouped_rmc = 0

    data_m_selected['temp1'] = 1
    grouped_itin_carrier_flight = data_m_selected.groupby(['ITINERARY_CARRIER'])['temp1'].sum()
    data_m_selected['FREQUENCY'] = data_m_selected['ITINERARY_CARRIER'].apply(lambda x: grouped_itin_carrier_flight.loc[x])

    data_c = pd.read_csv(get_path(year, quarter, 0, '.csv', select=False))

    data_m_selected = pd.merge(data_m_selected, data_c[['ITIN_ID', 'FARE_CLASS']], how='left', on=['ITIN_ID'])

    grouped_market_size = data_m_selected.groupby(['ITINERARY_NONDIRECTION'])['PASSENGERS'].sum()
    data_m_selected['MARKET_SIZE'] = data_m_selected['ITINERARY_NONDIRECTION'].apply(lambda x: grouped_market_size.loc[x])
    grouped_market_size = 0

    grouped_firm_size = data_m_selected.groupby(['REPORTING_CARRIER'])['PASSENGERS'].sum()
    data_m_selected['temp2'] = data_m_selected['REPORTING_CARRIER'].apply(lambda x: grouped_firm_size.loc[x])
    data_m_selected['FIRM_SIZE'] = data_m_selected['CARRIER_TOTAL_FLIGHTS'] - data_m_selected['temp2']
    grouped_firm_size = 0

    data_m_selected['temp1'] = 1
    grouped_carrier_origin_passengers = data_m_selected.groupby(['ORIGIN_CARRIER'])['temp1'].sum()
    grouped_carrier_dest_passengers = data_m_selected.groupby(['DEST_CARRIER'])['temp1'].sum()
    data_m_selected['ORIGIN_CARRIER_TOTAL_PASSENGERS'] = data_m_selected['ORIGIN_CARRIER'].apply(lambda x: grouped_carrier_origin_passengers[x])
    data_m_selected['DEST_CARRIER_TOTAL_PASSENGERS'] = data_m_selected['DEST_CARRIER'].apply(lambda x: grouped_carrier_dest_passengers[x])
    data_m_selected['ORIGIN_DEST_CARRIER_TOTAL_PASSENGERS'] = data_m_selected['ORIGIN_CARRIER_TOTAL_FLIGHTS'] + data_m_selected['DEST_CARRIER_TOTAL_FLIGHTS']
    grouped_carrier_origin_passengers, grouped_carrier_dest_passengers = 0, 0

    del data_m_selected['CARRIER_TOTAL_FLIGHTS']
    del data_m_selected['ORIGIN_CARRIER']
    del data_m_selected['DEST_CARRIER']
    del data_m_selected['ORIGIN_CARRIER_TOTAL_FLIGHTS']
    del data_m_selected['DEST_CARRIER_TOTAL_FLIGHTS']
    del data_m_selected['ORIGIN_CARRIER_TOTAL_PASSENGERS']
    del data_m_selected['DEST_CARRIER_TOTAL_PASSENGERS']

    return data_m_selected




def gen_firms_markets(data_m_selected):
    firms = list(set(list(data_m_selected['REPORTING_CARRIER'])))
    markets = list(set(list(data_m_selected['ITINERARY_NONDIRECTION'])))
    firms.sort()
    markets.sort()

    return markets, firms



def gen_fm_matrix(data, markets, firms, D, res_name):
    df = data_m_selected.drop_duplicates('ITINERARY_CARRIER').groupby('ITINERARY_CARRIER').sum()[['LEADER_' + D, 'CHALLENGER_' + D, 'INCUMBENT_imt']]
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

    leader = np.array(gen_fm_matrix(data_m_selected, markets, firms, D, 'LEADER'))
    challenger = np.array(gen_fm_matrix(data_m_selected, markets, firms, D, 'CHALLENGER'))
    incumbent = np.array(gen_fm_matrix(data_m_selected, markets, firms, D, 'INCUMBENT_imt'))

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




def last_process(data_m_selected, markets, firms, D):
    incumbent = np.array(gen_fm_matrix(data_m_selected, markets, firms, D, 'INCUMBENT_imt'))
    grouped_firm_MS = data_m_selected.drop_duplicates('REPORTING_CARRIER').groupby(['REPORTING_CARRIER'])['MS'].sum()
    grouped_itin_carrier_MS = data_m_selected.drop_duplicates('ITINERARY_CARRIER').groupby(['ITINERARY_CARRIER'])['MS'].sum()
    data_m_selected['AMS_OUTSIDE'] = data_m_selected['ITINERARY_CARRIER'].apply(lambda x: (grouped_firm_MS.loc[x.split('~')[1]] - grouped_itin_carrier_MS.loc[x]) / np.sum(incumbent[:, firms.index(x.split('~')[1])]))
    grouped_firm_MS, grouped_itin_carrier_MS = 0, 0

    return data_m_selected

## core program

if __name__ == '__main__':
    path = './industrial economy/'
    file = ['COUPON', 'MARKET', 'TICKET']
    time = '2017_01'
    year, quarter = time.split('_')
    D_list = ['MS', 'MD', 'RC']
    
    data_m_selected = selection(time)
    
    markets, firms = gen_firms_markets(data_m_selected)
    
    for D in D_list:
        data_m_selected = fill_mmc(data_m_selected, markets, firms, D)

    for D in D_list:
        data_m_selected = last_process(data_m_selected, markets, firms, D)

    data_m_selected.to_csv(path + 'selected/' + time + '/MARKET_' + time + '_selected.csv')
    



