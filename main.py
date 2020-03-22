# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 09:14:18 2020

@author: Nik
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import datetime
import os
import numpy as np

plt.close('all')

from scipy.stats import linregress

import data_helpers as dh

today = datetime.date.today()




#outliers = 

start_date = datetime.date(2020, 3, 2)
cntry = 'Australia'
state = 'Confirmed'

plt.close('all')
plt.ion()

this_loc = __file__
code_path = code_path = Path(this_loc).parents[1]
loc = code_path.joinpath('COVID-19',
                         'csse_covid_19_data',
                         'csse_covid_19_time_series',
                         )


csse_file = loc.joinpath(f'time_series_19-covid-{state}.csv')

#who_loc = code_path.joinpath('who_covid_19_situation_reports',
#                             'who_covid_19_sit_rep_time_series',
#                             )
#who_file = who_loc.joinpath('who_covid_19_sit_rep_time_series.csv')


# =============================================================================
# Create Total figures
# =============================================================================


data = pd.read_csv(loc.joinpath(f'time_series_19-covid-{state}.csv'))
df = data.set_index(list(data.columns[0:2])).drop(['Lat', 'Long'], axis=1)
df.columns = pd.to_datetime(df.columns, infer_datetime_format=True)
total = df.sum().reset_index().rename(columns={'index': 'Date', 0: 'Total'})
total = total.set_index('Date')

fig, ax = dh.create_plot(total)

# =============================================================================
# See Focus
# =============================================================================
fig, axs = plt.subplots(2,1)
focus = dh.country(df, cntry)

dh.create_plot(focus, ax=axs[0])
dh.create_plot(focus, logy=True, ax=axs[1])


# =============================================================================
# Create fittable regions
# =============================================================================
sub_focus = dh.fittable(focus)
#sub_focus = focus.iloc[-14:]
sub_focus = sub_focus[sub_focus.index >= pd.Timestamp(start_date)]

fig4, ax4 = dh.create_plot(sub_focus, logy=True)

xrange = np.array(range(len(sub_focus)))
N = np.log(dh.fittable(sub_focus)).squeeze()

mask = ~pd.isna(N)

# Outliers not included in fit.
outliers = pd.read_excel('Outliers.xlsx', index_col='Country')
if cntry in outliers.index:
    outl = outliers.loc[[cntry]]
    mask.loc[outl['Date']] = False
    print('No Outliers')

slope, intercept, rvalue, pvalue, stderr = linregress(x=xrange[mask],
                                                      y=N[mask],
                                                      )
sub_focus['Fit (exp)'] = np.exp(intercept)*np.exp(xrange*slope)


figf, axsf = plt.subplots(2, 1)

dh.create_plot(sub_focus, ax=axsf[0], logy=False)

dh.create_plot(sub_focus,  ax=axsf[1], logy=True)
