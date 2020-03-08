import sqlite3
# links: movieId,imdbId,tmdbId
# ratings: userId,movieId,rating,timestamp
# metas: imdb_id, title

con = sqlite3.connect('mydatabase.db')
cursorObj = con.cursor()
# cursorObj.execute("CREATE TABLE links(movieId text PRIMARY KEY, imdbId text, tmdbId text)")
# con.commit()
# cursorObj.execute("CREATE TABLE ratings(userId text PRIMARY KEY, movieId text, rating float, timestamp text)")
# con.commit()
# cursorObj.execute("CREATE TABLE metas(imdbid text PRIMARY KEY, title text)")
# con.commit()

cursorObj.execute("delete from links")
cursorObj.execute("INSERT INTO links VALUES ('8','8','80')")
con.commit()

links = [
    ("1", "1","1"),
    ("2", "1","1"),
    ("3", "1","1"),
]

cursorObj.execute("select * from links")
rows = cursorObj.fetchall()
for row in rows:
    print(row)


cursorObj.execute("INSERT INTO links(movieId, imdbId, tmdbId) VALUES (?,?,?)", [("3","3","3")])
con.commit()

cursorObj.execute("select * from links")
rows = cursorObj.fetchall()
for row in rows:
    print(row)

