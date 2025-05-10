import keyboard
from time import sleep
import os

import ctypes
winmm = ctypes.WinDLL('winmm')
winmm.timeBeginPeriod(1)


def checkstop():
    #Use in a while loop to be able to terminate easily. I know there are fancier ways to do this, but this works reliably
    if keyboard.is_pressed('end'):  # if key 'end' is pressed 
        print('STOPPING SIR at the end of this iteration, please be patient')
        return False
    else:
        return True

import time
def timestring():
    return time.strftime("%Y-%m-%d--%H-%M-%S")

import datetime
import logging

logger = logging.getLogger()

def setup_file_logger(log_file):
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)

def log(message):
    #outputs to Jupyter console
    print('{} {}'.format(datetime.datetime.now(), message))
    #outputs to file
    logger.info(message)