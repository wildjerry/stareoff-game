import sqlite3

def configure_db():
    db_connection = sqlite3.connect('stare.db')
    db_cursor = db_connection.cursor()
    #image-for image data as bytes|score-for how long the user lasted|uuid - universally unique identifier
    db_cursor.execute('CREATE TABLE Leaderboard(image, score, uuid)')

if __name__== '__main__':
    configure_db()