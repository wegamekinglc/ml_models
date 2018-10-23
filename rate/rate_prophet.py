# -*- coding: utf-8 -*-
"""
Created on 2018-10-16

@author: cheng.li
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy import signal
from scipy.interpolate import interp1d
from fbprophet import Prophet

plt.style.use("seaborn-poster")
rc('font', **{'family': 'Microsoft Yahei', 'size': 10})
rc('mathtext', **{'default': 'regular'})
rc('legend', **{'frameon': False})

df = pd.read_excel("../data/usd_rate.xls", index_col=0).resample("M").last().dropna() / 100.

ax = df.plot(cmap="gray")
ax.set_ylim((0., 0.16))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
ax.set_xlabel('')
plt.legend(["美国长期国债到期收益率"])

plt.savefig("usd_rate_history.svg", bbox_inches='tight')

m = Prophet(changepoint_range=1., changepoint_prior_scale=0.3, yearly_seasonality=False)
longest_cycle = 365 * 25
m.add_seasonality('25y', period=longest_cycle, fourier_order=int(longest_cycle / 90.))
train = df.reset_index()
m.fit(train)

future = m.make_future_dataframe(periods=0)
forecast = m.predict(future)
forecast['cycle'] = forecast['yhat'] - forecast['trend']
forecast['resid.'] = train['y'] - forecast['yhat']

fig, axes = plt.subplots(3, figsize=(18, 12))
forecast.set_index('ds')['trend'].plot(cmap='gray', ax=axes[0])
plt.ylim((0.0, 0.12))

forecast.set_index('ds')['cycle'].plot(cmap='gray', ax=axes[1])
plt.ylim((-0.03, 0.03))

forecast.set_index('ds')['resid.'].plot(cmap='gray', ax=axes[2])

formats = ['{:,.0%}', '{:,.1%}', '{:,.1%}']
ylabels = ['趋势', '周期', '残差']

for ax, f, ylabel in zip(axes, formats, ylabels):
    ax.spines['top'].set_visible(False)
    ax.set_yticklabels([f.format(x) for x in ax.get_yticks()])
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)

plt.grid(False)
plt.savefig("usd_rate_decompose.svg", bbox_inches='tight')

freqs, psd = signal.periodogram(forecast['cycle'])
spec = pd.Series(psd[1:], index=1. / freqs[1:] / 12).sort_index()
f = interp1d(spec.index, np.log(spec.values), kind='cubic', bounds_error=False)
xnew = np.linspace(spec.index[0], spec.index[-1], num=1000, endpoint=False)
ynew = f(xnew)

fig = plt.figure()
ax = fig.add_subplot(111)
spec = pd.Series(np.exp(ynew)[:-400], xnew[:-400])
spec.plot(cmap='gray', ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('')
plt.xlabel('频率（单位：年）')
plt.ylabel('谱能量（单位：1e-2）')
ax.set_yticklabels(['{:0.1f}'.format(x*100) for x in ax.get_yticks()])
plt.grid(False)
plt.savefig("usd_rate_spectral.svg", bbox_inches='tight')