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
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import data_helpers as dh

today = datetime.date.today()

def report_slope(slope, stderr, rvalue):
    
    double_time = np.log(2)/slope
    print(f'Fitted growth parameter = {slope:.3g} ± {stderr:.1g} (R^2={rvalue**2:.2g})')
    print(f'Doubling time ln2/alpha = {double_time:.2g}')


mask_outliers = True

start_date = pd.Timestamp.min
last_date = pd.Timestamp.max

start_mask = pd.Timestamp.min
last_mask = pd.Timestamp.max

# =============================================================================
# 
# =============================================================================
start_date = datetime.date(2021, 11, 1)

start_mask = datetime.date(2021, 12, 6)
last_mask = datetime.date(2021, 12, 21)

cntry = 'Australia'

state = False
state = 'New South Wales'

report = 'Confirmed'

plt.close('all')
plt.ion()

this_loc = __file__
code_path = code_path = Path(this_loc).parents[1]
loc = code_path.joinpath('COVID-19',
                         'csse_covid_19_data',
                         'csse_covid_19_time_series',
                         )


csse_file = loc.joinpath(f'time_series_covid19_{report}_global.csv')

# =============================================================================
# Create Total World figures
# =============================================================================


data = pd.read_csv(loc.joinpath(csse_file))
df = data.set_index(list(data.columns[0:2])).drop(['Lat', 'Long'], axis=1)
df.columns = pd.to_datetime(df.columns, infer_datetime_format=True)
df = df.dropna(axis=1, how='all')

total = df.sum().reset_index().rename(columns={'index': 'Date', 0: 'World'})
total = total.set_index('Date')

fig, ax = dh.create_plot(total)

# =============================================================================
# See Focus
# =============================================================================
fig, axs = plt.subplots(2, 1)

focus = dh.country(df, cntry, state=state)

cases_column = focus.columns[0]

dh.create_plot(focus, ax=axs[0])
dh.create_plot(focus, logy=True, ax=axs[1])


# =============================================================================
# Create fittable regions
# =============================================================================
sub_focus = dh.fittable(focus)
sub_focus['ElapsedDays'] = dh.get_elapsed_days(sub_focus)


sub_focus = sub_focus[sub_focus.index >= pd.Timestamp(start_date)]
sub_focus = sub_focus[sub_focus.index <= pd.Timestamp(last_date)]


fig4, ax4 = dh.create_plot(sub_focus[cases_column], logy=True)

xrange = np.arange(len(sub_focus))
# xrange = sub_focus['ElapsedDays'].values

cases = np.log(dh.fittable(sub_focus[cases_column])).squeeze()

mask = ~pd.isna(cases)

mask[mask.index <= pd.Timestamp(start_mask)] = False
mask[mask.index >= pd.Timestamp(last_mask)] = False

# Outliers not included in fit.
if mask_outliers:
    outliers = pd.read_excel('Outliers.xlsx', index_col='Country')
    if cntry in outliers.index:
        outl = outliers.loc[[cntry]]
        # Drop outliers if already not included in data
        mask.loc[mask.index.isin(outl['Date'])] = False
        
# =============================================================================
# Exponatial Model.
# =============================================================================

slope, intercept, rvalue, pvalue, stderr = linregress(x=xrange[mask],
                                                      y=cases[mask],
                                                      )
sub_focus['Fit (exp)'] = np.exp(intercept)*np.exp(xrange*slope)

# =============================================================================
# Expontial Growth
# =============================================================================
X = xrange[mask].reshape(-1, 1)
y = cases[mask].values.reshape(-1, 1)

reg_exp = LinearRegression().fit(X, y)

def scipy_return(model, X, y):
    """Mimic Scipy stats equivalent parameter return."""
    slope = model.coef_.squeeze()
    intercept = model.intercept_.squeeze()
    r2 = reg_exp.score(X, y) # TODO wrapped for generalisation.
    # TODO this needs to generalise to correlation matrix
    rvalue = np.sign(slope)*r2**0.5
    return slope, intercept, rvalue, None, None
    
slope_sk, intercept_sk, rvalue_sk, pvalue_sk, stderr_sk = scipy_return(reg_exp,
                                                                       X, y)
    
# =============================================================================
# Exponatial Growth with increaseing linear.
# =============================================================================
square_feature = PolynomialFeatures(2).fit_transform(X)[:, 1:] #Drop intercept.
reg_k_linear = LinearRegression().fit(square_feature, y)


sub_focus['Fit (exp(R(t))'] = np.exp(reg_k_linear.intercept_)*np.exp(xrange*slope)

figf, axsf = plt.subplots(4, 1, figsize=(6.4, 9.6))
figf.suptitle('Cases vs Models')

fitted_cases = sub_focus[[cases_column, "Fit (exp)"]]

dh.create_plot(fitted_cases, ax=axsf[0], logy=False, title='Total Cases')
dh.create_plot(fitted_cases, ax=axsf[1], logy=True)
dh.create_plot(fitted_cases - fitted_cases.shift(1),
                ax=axsf[2], logy=False, title='Daily Cases')
dh.create_plot(fitted_cases - fitted_cases.shift(1),
                ax=axsf[3], logy=True)

fig_residual, axs_res = plt.subplots()
fig_residual.suptitle('Models Residuals')

residual = 100*(sub_focus[cases_column] - sub_focus['Fit (exp)'])/sub_focus['Fit (exp)']
residual = residual.to_frame()
residual.columns = ['Diff %']

dh.create_plot(residual, ax=axs_res)

residual.hist()
ax_hist = residual[mask].hist()

n_data = len(residual[mask])
res_mean = residual.mean()
res_std = residual.std()
res_stderr = res_std/np.sqrt(n_data)

interval = ax_hist[0][0].xaxis.get_data_interval()
xspace = np.linspace(interval[0], interval[1])
ax_hist[0][0].plot(xspace, norm(loc=res_mean, scale=res_std).pdf(xspace))


report_slope(slope, stderr, rvalue)
report_slope(slope_sk, np.inf, rvalue_sk)
