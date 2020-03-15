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

from scipy.stats import linregress

today = datetime.date.today()


def create_plot(df, week_marks=True, **kwargs):
    
    fig, ax = plt.subplots()
    
    df.plot(ax=ax, **kwargs)
    
    if week_marks:
        for date in focus.index.array[6::7]:
            date = pd.Timestamp(date)
            ax.axvline(x=date, color='k',
                       linestyle='--',
                       alpha=0.5,
                       lw=0.5)
            
        for date in focus.index.array[1:]:
            date = pd.Timestamp(date)
            ax.axvline(x=date, color='k',
                       linestyle='--',
                       alpha=0.2,
                       lw=0.3)
    
    return fig, ax

def fittable(df, least_cases=1):
    return df.iloc[np.where(df>=1)[0]]

def create_plotBK(df, week_marks=True, month_mark=False):
    
    pass

def get_save_loc(today):
    save_place = Path(os.path.join('results', str(today)))
    save_place.absolute().mkdir(parents=True, exist_ok=True)
    return save_place

def day_name(date):
    
    day = today.weekday()
    names = {0: 'Monday',
             1: 'Tuesday',
             2: 'Wednesday',
             3: 'Thursday',
             4: 'Friday',
             5: 'Satday',
             6: 'Sunday'}
    return names[day]

def make_folder(path):
    
    if not os.path.exists(path):
        os.mkdir(path)

def growth(df):
    return df-df.shift(1)

def growth_rate(df):
    return growth(df)/df.shift(1)

def growth_factor(df):
    return growth(df) / growth(df).shift(1)

def country(df, cntry, states=False):
    country = df.xs(cntry, level=1)
    if not states:
   
        country = pd.DataFrame({cntry: country.sum()})

    country.index.name = 'Date'

    return country



cntry = 'Australia'

plt.close('all')
plt.ion()

loc = r'C:\RWA\Code\COVID-19\csse_covid_19_data\csse_covid_19_time_series'
loc = Path(loc)

state = 'Confirmed'

data = pd.read_csv(loc.joinpath(f'time_series_19-covid-{state}.csv'))

df = data.set_index(list(data.columns[0:2])).drop(['Lat', 'Long'], axis=1)
df.columns = pd.to_datetime(df.columns, infer_datetime_format=True)

total = df.sum().reset_index().rename(columns={'index': 'Date', 0: 'Total'})

total = total.set_index('Date')
ax = total.plot()

focus = country(df, cntry)

#plt.figure()
#ax2 = focus.plot()
#fig = ax2.figure


#plt.ioff()



sub_focus = fittable(focus).iloc[-14:]
#sub_focus = total.iloc[5:-30]


country_list = data['Country/Region'].unique()

fittable(sub_focus).plot(logy=True)

fig3, ax3 = create_plot(sub_focus, logy=True)

xrange = np.array(range(len(sub_focus)))
N = np.log(fittable(sub_focus)).squeeze()

mask = ~pd.isna(N)
#mask.iloc[-3]=False

slope, intercept, rvalue, pvalue, stderr = linregress(x=xrange[mask],
                                                      y=N[mask],
                                                      )

sub_focus['Fit']= np.exp(intercept)*np.exp(xrange*slope-stderr)

fig4, ax4 = create_plot(sub_focus, logy=False)

