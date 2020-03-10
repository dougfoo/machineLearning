import sqlite3
import query
from query import timeit

# links: movieId,imdbId,tmdbId
# ratings: userId,movieId,rating,timestamp
# metas: imdb_id, title
cursorObj = None
con = None

@timeit
def create(memory=False):
    global con 
    global cursorObj 
    if (memory):
        print('memory only')
        con = sqlite3.connect(':memory:')
        cursorObj = con.cursor()
    else:
        print('mydatabase.db')
        con = sqlite3.connect('mydatabase.db')
        cursorObj = con.cursor()
        cursorObj.execute("DROP TABLE links") 
        cursorObj.execute("DROP TABLE metas")
        cursorObj.execute("DROP TABLE ratings")

    cursorObj.execute("CREATE TABLE links(movieId text PRIMARY KEY, imdbId text, tmdbId text)")
    cursorObj.execute("CREATE TABLE ratings(userId text, movieId text, rating float, timestamp text)")
    cursorObj.execute("CREATE TABLE metas(imdbid text, title text)")

    files = ["links.csv", "ratings.csv","movies_metadata.csv"]
    links = query.load_file(files[0])[1]
    ratings = query.load_file(files[1])[1]
    metas = query.load_file(files[2])[1]

    insert("INSERT INTO links(movieId, imdbId, tmdbId) VALUES (?,?,?)", links)
    select("links")

    insert("INSERT INTO ratings(userId, movieId, rating, timestamp) VALUES (?,?,?,?)", ratings)
    select("ratings")

    insert_metas("INSERT INTO metas(imdbId, title) VALUES (?,?)", metas)
    select("metas")

@timeit
def delete():
    cursorObj.execute("delete from links")
    con.commit()

@timeit
def insert(sql, data):
    for i, row in enumerate(data):
        cursorObj.execute(sql, row)
        if (i % 500 == 0):
            con.commit()
    con.commit()

@timeit
def insert_metas(sql, data):
    for i, row in enumerate(data):
        cursorObj.execute(sql, (row[6],row[20]))
        if (i % 500 == 0):
            con.commit()
    con.commit()

def select(table):
    cursorObj.execute("select count(*) from "+table)
    rows = cursorObj.fetchall()
    for row in rows:
        print(table + ' count: '+str(row))

@timeit
def index():
    cursorObj.execute("create index lidx on links(imdbId, movieId)")
    cursorObj.execute("create index ridx on ratings(movieId, rating)")
    cursorObj.execute("create index midx on metas(imdbId, title)")

@timeit
def join():
    #cursorObj.execute("select * from links l, ratings r, metas m where l.movieId=r.movieId and l.imdbId=m.imdbId")
    cursorObj.execute("select m.title, avg(r.rating) avg_rating, count(r.rating) from links l, ratings r, metas m where l.movieId=r.movieId and l.imdbId=m.imdbId group by m.title having count(r.rating) > 2 and avg_rating > 4.5 ")
    rows = cursorObj.fetchall()
    for c,row in enumerate(rows):
        print('sql join count: '+str(row))
        if (c > 5):
            print('count: '+str(len(rows)))
            break

if __name__ == "__main__":
    create(memory=False)
    index()
    join()
    join()
