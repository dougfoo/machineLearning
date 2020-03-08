import os
import csv 
from functools import wraps
from time import time
from typing import Tuple
import pandas as pd

def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def print_head(rows):
    for row in rows[:5]: 
        print("%s %s %s %s %s"%(row[0],row[1],row[2],row[3],row[4])), 

@timeit
def load_file(file) -> []:
    fields,rows = [],[]     
    with open(file, 'r', encoding='utf8') as csvfile: 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)    # strip header
        for row in csvreader: 
            rows.append(row) 
        print("File [%s], Total no. of rows: %d"%(file, csvreader.line_num)) 
    return  fields, rows

@timeit
def load_df(file) -> pd.DataFrame:
    return pd.read_csv(file)

@timeit
def merge(links, ratings, metas):
    merged = []
    for link in links[1]:
        mlink = link.copy()
        mlink += (['',''])  # rating and name
        for rating in ratings[1]:
            if (mlink[0] == rating[1]):
                mlink[3] = rating[2]
                break
        for meta in metas[1]:
            if (mlink[1] == meta[6]):  # stripped tt off meta imdb columns FYI
                mlink[4] = meta[20]
                break
        merged.append(mlink)
    return merged

@timeit
def make_map(tab, index):
    m = {}
    for t in tab:
        m[t[index]] = t
    return m

@timeit
def merge_wmap(links, ratings, metas) -> []:
    ratings_map = make_map(ratings[1], 1)
    metas_map = make_map(metas[1], 6)
    merged = []
    for link in links[1]:
        mlink = link.copy() + ['','']
        mlink[3] = ratings_map.get(link[0])[2] if ratings_map.get(link[0]) else ''
        mlink[4] = metas_map.get(link[1])[20] if metas_map.get(link[1]) else '' 
        merged.append(mlink)
    return merged

@timeit
def pandas_join(links_df, ratings_df, metas_df) -> pd.DataFrame:
    merged_df = pd.merge(links_df[['movieId','imdbId']], ratings_df[['movieId','rating']], on='movieId', how='right', validate='one_to_many')
    merged_df = pd.merge(merged_df, metas_df[['title','imdb_id']], left_on='imdbId', right_on='imdb_id', how='left', validate='many_to_many')
    return merged_df

def sqlite_load():
    pass

def sqlite_join():
    pass

files = ["links.csv", "ratings.csv","movies_metadata.csv"]

# load files
links = load_file(files[0])
ratings = load_file(files[1])
metas = load_file(files[2])

# manual merge -- takes a LONG time
# newlinks = merge(links, ratings, metas)
# print_head(newlinks)

# manual w/ maps
newlinks_wmap = merge_wmap(links, ratings, metas)
print_head(newlinks_wmap)
print(len(newlinks_wmap))

# load pandas dfs
links_df = load_df(files[0])
ratings_df = load_df(files[1])
metas_df = load_df(files[2])

# using pandas merge
merged_df = pandas_join(links_df, ratings_df, metas_df)
print(merged_df.head())
print(merged_df.shape)

# using sqlite


