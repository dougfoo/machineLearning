import csv
import datetime
import json
import pandas as pd
import re
import time
import requests
import argparse
import warnings


def _price_to_int(s):
    if (isinstance(s,list)):    # sloppy hack
        s = s[0]
    return int(re.sub('[$,]', '', s))

# It turns out Blue Nile's API will only let us grab 1000 diamonds
# associated with each query. To get around this, let's use price to
# page through the results.
# 1. Get first thousand diamonds in query.
# 2. Use diamond with highest price in that query to seed the next
#    query.
def diamonds(params):
    assert params['sortColumn'] == 'price' and params['sortDirection'] == 'asc'

    landing_page = requests.get('http://www.bluenile.com/')
    url = 'http://www.bluenile.com/api/public/diamond-search-grid/v2'
    result = []
    counter = 0
    restartCounter = 0
    while True:
        try:
            response = requests.get(url, params, cookies=landing_page.cookies)
        except:
            print ('caught major fetch exception', datetime.datetime.now())
            time.sleep(60 * 30)  # n-min per iteration
            next

        print ('response',response, response.headers)
        print (response.request.url)

        if (response.ok == False):
            print('response failed, retry same again after big wait', datetime.datetime.now())
            with open('interim-result'+str(restartCounter)+'.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(result[0].keys()) # header
                for row in result:                 # details
                    writer.writerow(row.values())
            print('writeFile',f)
            time.sleep(60 * 30)  # n-min per iteration after a fail
            restartCounter += 1
            next

        d = json.loads(response.text)
        last_page = params['pageSize'] >= d['countRaw']

        for i in range(len(d['results'])):
            d['results'][i]['price'] = _price_to_int(d['results'][i]['price'])
        max_price = d['results'][-1]['price']
        min_price = d['results'][0]['price']

        if last_page:
            result += d['results']
            with open('result.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(result[0].keys()) # header
                for row in result:                 # details
                    writer.writerow(row.values())
            print('writeFile',f)
            break
        else:
            assert min_price < max_price, 'There are over %d diamonds with these characteristics at this price %d.' % (params['pageSize'], min_price)
            result += [x for x in d['results'] if x['price'] < max_price]
            params['minPrice'] = max_price

            with open('result'+str(counter)+'.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(result[0].keys())  # header
                for row in result:                 # details
                    writer.writerow(row.values())
            print('writeFile: ',writeFile.name)
            counter += 1 
            print ('sleeping ...  for next loop ', datetime.datetime.now())
            time.sleep(60 * 4)  # n-min per iteration
    return result

def clean(data):
    # Put the data into a data frame.
    df = pd.DataFrame(data)

    # # Clean up the data.
    # for col in ['carat', 'depth', 'lxwRatio', 'table']:
    #     if (isinstance(df[col],list)):   # sloppy hack
    #         df[col] = df[col][0]
    #     print(col,df[col])
    #     df[col] = df[col].map(lambda s: s.replace(',', '')).astype(float)
    # for col in ['pricePerCarat']:
    #     if (isinstance(df[col],list)):   # sloppy hack
    #         df[col] = df[col][0]
    #     df[col] = df[col].map(_price_to_int)

    return df


def parse_arguments():
    parser = argparse.ArgumentParser()

    shapes = [
        "RD",  # round
        "PR",  # princess
        "EC",  # emerald
        "AS",  # asscher
        "CU",  # cushion
        "MQ",  # marquise
        "RA",  # radiant
        "OV",  # oval
        "PS",  # pear
        "HS",  # heart
    ]
    parser.add_argument('--shape', nargs='+', choices=shapes)

    parser.add_argument('--minPrice', type=int)
    parser.add_argument('--maxPrice', type=int)
    parser.add_argument('--minCarat', type=float)
    parser.add_argument('--maxCarat', type=float)

    cuts = ['Good', 'Very Good', 'Ideal', 'Signature Ideal']
    colors = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarities = ['SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']
    select_one = [
        ('minCut', cuts),
        ('maxCut', cuts),
        ('minColor', colors),
        ('maxColor', colors),
        ('minClarity', clarities),
        ('maxClarity', clarities),
    ]
    for key, choices in select_one:
        parser.add_argument('--%s' % key, choices=choices)

    arguments_with_defaults = [
        ('startIndex', 0),
        ('pageSize', 1000),
        ('country', 'USA'),
        ('language', 'en-us'),
        ('currency', 'USD'),
        ('sortColumn', 'price'),
        ('sortDirection', 'asc'),
    ]
    for k, v in arguments_with_defaults:
        parser.add_argument('--%s' % k, default=v, type=type(v))

    args = parser.parse_args()
    d = args.__dict__
    for k in list(d):
        if d[k] is None:
            del d[k]
    return d


def main():
    params = parse_arguments()
    l = diamonds(params)
    df = clean(l)
    print (df.head())
    print (df.to_csv(path_or_buf='./out.csv', index=False))

if __name__ == '__main__':
    main()
