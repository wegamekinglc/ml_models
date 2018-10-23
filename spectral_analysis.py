# -*- coding: utf-8 -*-
"""
Created on 2018-10-16
@author: cheng.li
"""

import pandas as pd
from fbprophet import Prophet
from scipy import signal
from matplotlib import pyplot as plt


name = "gdp"
file_name = "data/usd_{0}.xls".format(name)
freq = "Q"
periods = 12 if freq == 'M' else 4
df = pd.read_excel(file_name, index_col=0).resample(freq).last().dropna()
df = df.reset_index()

m = Prophet(changepoint_range=1., changepoint_prior_scale=0.2)
m.fit(df)
future = m.make_future_dataframe(periods=0)
forecast = m.predict(future)
de_trend = df['y'].values - forecast['trend'].values

fig, axes = plt.subplots(3, 1, figsize=(14, 7))

axes[0].plot(df.ds, forecast['trend'])
axes[0].set_title('US {0} trend part'.format(name))

axes[1].plot(df.ds, df['y'] - forecast['trend'])
axes[1].set_title('US {0} de-trend part'.format(name))

freqs, psd = signal.welch(de_trend)
axes[2].plot(1. / freqs / periods, psd)
axes[2].set_title('PSD: power spectral density for US {0}'.format(name))
axes[2].set_xlabel("Year")
plt.tight_layout()

fig.savefig("{0}.png".format(name), dpi=800)
