# -*- coding: utf-8 -*-
"""
Created on 2018-10-22

@author: cheng.li
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import rc
from matplotlib import pyplot as plt

plt.style.use("seaborn-poster")
rc('font', **{'family': 'Microsoft Yahei', 'size': 10})
rc('mathtext', **{'default': 'regular'})
rc('legend', **{'frameon': False})

df = pd.read_excel("../data/usd_gdp.xls", index_col=0).resample("Q").last().dropna() / 100.
split_point = len(df)

order = [2, 0, 2]

n = len(df)

df['model'] = np.nan
f_periods = 24
endog = df[:n-f_periods]['y'].values

model = ARIMA(endog, order=order).fit()
df['model'][:n-f_periods] = model.predict()
df['model'][n-f_periods:] = model.forecast(f_periods)[0]


fig = plt.figure()
ax = fig.gca()
plt.set_cmap('gist_gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
df['y'].plot(linestyle="-", ax=ax, cmap="gray")
df['model'].plot(linestyle="-.", ax=ax, cmap="gray")
ax.set_xlabel('')
ax.set_yticklabels(['{:0.1f}%'.format(x*100) for x in ax.get_yticks()])
plt.legend(['真实值', '模型值'], ncol=2)
ax.axvline(x=df.index[-f_periods+1], linewidth=3, linestyle='--', color='grey')
ax.text(df.index[100], 0.10, '训练期', fontdict={'size': 15})
ax.text(df.index[270], 0.10, '预测期', fontdict={'size': 15})

plt.grid(False)
plt.savefig("usd_gdp_forecast_arima.svg", bbox_inches='tight')
plt.show()