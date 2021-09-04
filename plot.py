import plotly.graph_objects as go


def plot_anomalies(data, anomaly_indices, anomalies):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(len(data))], y=data,
                             mode='lines',
                             name='data'))

    if not anomalies.size == 0:
        fig.add_trace(go.Scatter(x=anomaly_indices, y=anomalies,
                                 mode='markers',
                                 name='anomalies'))
    fig.show()
