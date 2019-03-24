# get a list of trades
import from feature_functions *
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from plotly import tools



df = pd.read_csv("EURUSD.csv")
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

df.date = pd.to_datetime(df.date, format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df.drop_duplicates(keep=False)

ma = df.close.rolling(center=False, window=30).mean()

trace0 = go.Ohlc(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close,
                name='Currecny Quete')
trace1 = go.Scatter(x=df.index, y = ma)
trace2 = go.Bar(x=df.index, y=df.volume)


data = [trace0, trace1, trace2]

fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True)
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)


py.offline.plot(data, filename='usdeur.html')
print(df)


