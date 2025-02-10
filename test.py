import dlib
print(dlib.DLIB_USE_CUDA);

import cv2
import dlib
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from imutils import face_utils
import time
import sqlite3
from uuid import uuid4
def db():
    db_connection = sqlite3.connect('stare.db')
    db_cursor = db_connection.cursor()

    sqlite_query = '''
        WITH Top5 AS (
                SELECT * 
                FROM Leaderboard 
                ORDER BY score DESC 
                LIMIT 5
            )
            SELECT * FROM Top5
        '''
    db_cursor.execute(sqlite_query)
    leaderboard = db_cursor.fetchall()

    print(len(leaderboard))

def mpl_backends():
    for be in matplotlib.backends.backend_registry.list_builtin(matplotlib.backends.BackendFilter.INTERACTIVE):
        if matplotlib.rcsetup.validate_backend(be):
            print(f'interactive: {be}')

    for be in matplotlib.backends.backend_registry.list_builtin(matplotlib.backends.BackendFilter.NON_INTERACTIVE):
        if matplotlib.rcsetup.validate_backend(be):
            print(f'non interactive: {be}')

mpl_backends()

import cv2
    
    
def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    print('available\n', available_ports)
    print('working\n', working_ports)
    print('non_working\n', non_working_ports)
    return available_ports,working_ports,non_working_ports