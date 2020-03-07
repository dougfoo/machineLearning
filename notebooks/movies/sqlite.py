import sqlite3

con = sqlite3.connect('mydatabase.db')
cursorObj = con.cursor()
cursorObj.execute("CREATE TABLE employees(id integer PRIMARY KEY, name text, salary integer)")
con.commit()


cursorObj.execute("INSERT INTO employees VALUES(1, 'John', 700)")
cursorObj.execute("INSERT INTO employees VALUES(2, 'Mary', 800)")
cursorObj.execute("INSERT INTO employees VALUES(3, 'Mason', 900)")
con.commit()


