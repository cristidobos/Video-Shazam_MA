import sqlite3
import io
from sqlite3 import Error
import numpy as np


# Opens connection to database at specified path
def create_database(path):
    connection = None
    try:
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)
        connection = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
        print("Database created successfully!")
    except Error as e:
        print("Database could not be created. Error: \n{}".format(e))

    return connection


# Creates table for storing the video features
def create_table(connection):
    # Create a cursor object
    cursor = connection.cursor()

    # Delete table if already exists
    cursor.execute("DROP TABLE IF EXISTS Video")

    # Add table for storing videos
    cursor.execute('''
                    CREATE TABLE Video(
                    id INTEGER PRIMARY KEY ,
                    name TEXT NOT NULL,
                    mfcc array NOT NULL,
                    audio array NOT NULL,
                    colhist array NOT NULL,
                    tempdiff array NOT NULL,
                    chdiff array NOT NULL)'''
                   )

    # cursor.execute('''CREATE TABLE Video(id INTEGER PRIMARY KEY , name TEXT)''')

    # Commit changes
    connection.commit()
    print("Table created!")
    cursor.close()


def add_video_descriptor(id, name, descriptor, connection):
    # Create cursor object
    cursor = connection.cursor()

    # Add entry to database
    query = 'INSERT INTO Video(id, name, mfcc, audio, colhist, tempdiff, chdiff) VALUES (?,?,?,?,?,?,?)'

    cursor.execute(query, (id, name, descriptor['mfcc'], descriptor['audio'], descriptor['colhist'],
                           descriptor['tempdiff'], descriptor['chdiff']))

    connection.commit()

    # Return the id of the entry
    return cursor.lastrowid

def fetch_video_entry_by_id(id, connection):
    # Create cursor object
    cursor = connection.cursor()

    # Fetch entry from database
    cursor.execute('SELECT * FROM Video WHERE id = ?', (id, ))

    _, name, mfcc, audio, colhist, tempdiff, chdiff = cursor.fetchone()

    # Define descriptor and add items
    descriptor = {}
    descriptor['mfcc'] = mfcc
    descriptor['audio'] = audio
    descriptor['colhits'] = colhist
    descriptor['tempdiff'] = tempdiff
    descriptor['chdiff'] = chdiff

    connection.commit()

    return name, descriptor


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
