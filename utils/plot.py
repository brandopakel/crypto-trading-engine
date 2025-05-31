import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from plotly.subplots import make_subplots

def plot_strategy(coin : pd.DataFrame, title : str = "Strategy Visualization", overlays: Optional[list] = None, indicators: Optional[list] = None, signal_col: str = "signal") -> go.Figure:
    coin = coin.copy()    
    coin['timestamp'] = pd.to_datetime(coin['start'], unit='s')

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.1,row_heights=[0.7,0.3],specs=[[{'type':'xy'}],[{'type':'xy'}]])

    fig.add_trace(trace=go.Candlestick(
        x=coin['timestamp'], 
        open=coin['open'].astype(float), 
        high=coin['high'].astype(float), 
        low=coin['low'].astype(float), 
        close=coin['close'].astype(float), 
        name='Candlestick'
        ), row=1,col=1)

    if overlays:
        for col in overlays:
            if col in coin.columns:
                fig.add_trace(go.Scatter(
                    x = coin['timestamp'],
                    y = coin[col].astype(float),
                    mode='lines',
                    name=col
                ), row=1, col=1)

    
    if signal_col in coin.columns:
        buys = coin[coin[signal_col] == 1]
        sells = coin[coin[signal_col] == -1]

        fig.add_trace(go.Scatter(
            x=buys['timestamp'],
            y=buys['close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='lime', size=10)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sells['timestamp'],
            y=sells['close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ), row=1, col=1)

    if indicators:
        for col in indicators:
            if col in coin.columns:
                fig.add_trace(
                    go.Scatter(
                        x=coin['timestamp'],
                        y=coin[col].astype(float),
                        mode='lines',
                        name=col
                    ), row=2, col=1)


    fig.update_layout(xaxis_rangeslider_visible = False, title=title, xaxis_title="Time", yaxis_title="Price", template="plotly_dark", yaxis=dict(autorange=True), yaxis2=dict(autorange=True), showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), height = 800)
    fig.show()
    return fig