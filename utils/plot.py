import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from plotly.subplots import make_subplots

def plot_strategy(coin : pd.DataFrame, title : str = "Strategy Visualization", overlays: Optional[list] = None, indicators: Optional[list] = None, fib_levels: Optional[dict] = None, wave_labels: Optional[list] = None, signal_col: str = 'signal') -> go.Figure:
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

    if signal_col and signal_col in coin.columns:
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
    
    if fib_levels:
        for label, level in fib_levels.items():
            fig.add_hline(
                y=level,
                line=dict(color='gold',width=1,dash='dot'),
                annotation_text=label,
                annotation_position='right',
                row=1, col=1
            )

    if wave_labels:
        price_range = coin['close'].max() - coin['close'].min()
        offset_unit = price_range * 0.01

        color_map = {
            '1': 'orangered',
            '2': 'darkorange',
            '3': 'dodgerblue',
            '4': 'mediumpurple',
            '5': 'deepskyblue',
            'a': 'hotpink',
            'b': 'limegreen',
            'c': 'magenta'
        }

        for label in wave_labels:
            label_col = f"ew_{label}"
            if label_col in coin.columns:
                wave_points = coin[~coin[label_col].isna()].copy()
                if wave_points.empty:
                    continue

                y_offset = wave_labels.index(label) * offset_unit
                wave_points['offset_y'] = wave_points[label_col] + y_offset

                # 1. Add markers + text (with legend)
                fig.add_trace(go.Scatter(
                    x=wave_points['timestamp'],
                    y=wave_points['offset_y'],
                    mode='markers+text',
                    text=[label] * len(wave_points),
                    textposition='top center',
                    name=f'EW {label}',
                    marker=dict(size=10, symbol='circle', color=color_map.get(label, 'white')),
                    showlegend=True
                ), row=1, col=1)

                # 2. Add internal lines (no legend)
                if len(wave_points) >= 2:
                    fig.add_trace(go.Scatter(
                        x=wave_points['timestamp'],
                        y=wave_points['offset_y'],
                        mode='lines',
                        line=dict(color=color_map.get(label, 'white'), width=1.5),
                        name='',  # don't name it
                        showlegend=False
                    ), row=1, col=1)
    
    #For testing:
    #wave_cols = [col for col in coin.columns if col.startswith("ew_")]
    #print(coin[wave_cols].dropna(how='all').head(10))

    fig.update_layout(xaxis_rangeslider_visible = False, title=dict(text=title, y=0.92), xaxis_title="Time", yaxis_title="Price", template="plotly_dark", yaxis=dict(autorange=True), yaxis2=dict(autorange=True), showlegend=True, legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,  # push legend below plot
            xanchor='right',
            x=1
        ), height = 800)
    fig.show()
    return fig