# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:09:49 2020

@author: Nik
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import datetime
import os
import numpy as np


from scipy.stats import linregress


def create_plot(df, week_marks='Sun', ax=None, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    df.plot(ax=ax, **kwargs)
    
    if week_marks:
        for date in df.index.array[6::7]:
            date = pd.Timestamp(date)
            ax.axvline(x=date, color='k',
                       linestyle='--',
                       alpha=0.5,
                       lw=0.5)
            
        for date in df.index.array[1:]:
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
    
    day = date.weekday()
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