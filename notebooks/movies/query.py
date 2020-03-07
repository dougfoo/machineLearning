import os
import csv 
from functools import wraps
from time import time

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
        print("%s %s %s"%(row[0],row[1],row[2])), 

@timeit
def load_file(file):
    fields,rows = [],[]     
    with open(file, 'r', encoding='utf8') as csvfile: 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)    # strip header
        for row in csvreader: 
            rows.append(row) 
        print("File [%s], Total no. of rows: %d"%(file, csvreader.line_num)) 
    return  fields, rows

@timeit
def merge(links, ratings, metas):
    for link in links[1]:
        link += (['',''])  # rating and name
        for rating in ratings[1]:
            if (link[0] == rating[1]):
                link[3] = rating[2]
                break
        for meta in metas[1]:
            if (link[1] == meta[6]):  # stripped tt off meta imdb columns FYI
                link[4] = meta[20]
                break
    return links

@timeit
def make_map(tab, index):
    m = {}
    for t in tab:
        m[t[index]] = t
    return m

@timeit
def merge_wmap(links, ratings, metas):
    ratings_map = make_map(ratings[1], 1)
    metas_map = make_map(metas[1], 6)
    for link in links[1]:
        link += (['',''])  # rating and name
        link[3] = ratings_map.get(link[0],'')
        link[4] = metas_map.get(link[1],'')
    return links

files = ["links.csv", "ratings_small.csv","movies_metadata.csv"]

links = load_file(files[0])
ratings = load_file(files[1])
metas = load_file(files[2])

newlinks = merge(links, ratings, metas)
newlinks_wmap = merge_wmap(links, ratings, metas)

print_head(newlinks)
print_head(newlinks_wmap)

    # print('Field names for file >>'+ file +'<< are:' + ', '.join(field for field in fields)) 
    # print_head(rows)


