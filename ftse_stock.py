import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yahoo_finance
from yahoo_finance import Share
import numpy as np
import datetime as dt

FTSE100 = Share('^FTSE')

FTSE100hist = FTSE100.get_historical('2014-01-01', '2015-01-01')

Date = [dt.datetime.strptime(d['Date'], '%Y-%m-%d').date() for d in FTSE100hist]
Open = [d['Open'] for d in FTSE100hist]
Close = [d['Close'] for d in FTSE100hist]

# plt.plot_date(Date, Open)
# plt.plot_date(Date, Close)
# plt.show()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(Date,Open)
plt.plot(Date,Close)
plt.gcf().autofmt_xdate()
plt.show()