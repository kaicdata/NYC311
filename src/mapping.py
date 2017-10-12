# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:24:41 2017

@author: kaicui
"""


import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize


fig, ax = plt.subplots(figsize=(10,20))

#
# westlimit=-74.126129; southlimit=40.542504; eastlimit=-73.618011; northlimit=40.938934
m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
            lat_0=40.740719, lon_0=-73.87207,
            llcrnrlon=-74.126129, llcrnrlat= 40.542504, urcrnrlon=-73.618011, urcrnrlat=40.938934)

m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines()


m.readshapefile('data/other/nybb_17c/nybb', 'a')


m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
            lat_0=54.5, lon_0=-4.36,
            llcrnrlon=-6., llcrnrlat= 49.5, urcrnrlon=2., urcrnrlat=55.2)