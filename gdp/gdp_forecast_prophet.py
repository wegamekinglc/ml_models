# -*- coding: utf-8 -*-
"""
Created on 2018-10-23

@author: cheng.li
"""

import pandas as pd
from matplotlib import rc
from fbprophet import Prophet
from matplotlib import pyplot as plt

plt.style.use("seaborn-poster")
rc('font', **{'family': 'Microsoft Yahei', 'size': 10})
rc('mathtext', **{'default': 'regular'})
rc('legend', **{'frameon': False})

df = pd.read_excel("../data/usd_gdp.xls", index_col=0).resample("Q").last().dropna() / 100.

m = Prophet(changepoint_range=1., changepoint_prior_scale=0.3, yearly_seasonality=False)
longest_cycle = 365 * 25
m.add_seasonality('25y', period=longest_cycle, fourier_order=int(longest_cycle / 365))

f_periods = 24
train = df[:-f_periods].reset_index()
m.fit(train)

future = m.make_future_dataframe(periods=f_periods, freq='Q')
forecast = m.predict(future)
forecast.set_index('ds', inplace=True)
df['yhat'] = forecast['yhat']

fig = plt.figure()
ax = fig.gca()
plt.set_cmap('gist_gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
df['y'].plot(linestyle="-", ax=ax, cmap="gray")
df['yhat'].plot(linestyle="-.", ax=ax, cmap="gray")
ax.set_xlabel('')
ax.set_yticklabels(['{:0.1f}%'.format(x*100) for x in ax.get_yticks()])
plt.legend(['真实值', '模型值'], ncol=2)
ax.axvline(x=df.index[-f_periods+1], linewidth=3, linestyle='--', color='grey')
ax.text(df.index[100], 0.10, '训练期', fontdict={'size': 15})
ax.text(df.index[270], 0.10, '预测期', fontdict={'size': 15})

plt.grid(False)
plt.savefig("usd_gdp_forecast_prophet.svg", bbox_inches='tight')
plt.show()