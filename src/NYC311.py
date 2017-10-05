# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 10:57:04 2017
"""

import pandas as pd
import numpy as np
import path
import os
import time
import datetime

import common as com

ROOT_DIR = "C:\\Disks\\D\\Coatue\\NYC311\src"
#ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(ROOT_DIR)

data = com.import_data()
summary_data = com.summary_data(data)

data.to_csv('./output/data_nyc311.csv')
