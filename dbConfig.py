import sqlite3

db_connection = sqlite3.connect('stare.db')
db_cursor = db_connection.cursor()
db_cursor.execute('CREATE TABLE leaderboard(image, score)')