import matplotlib.pyplot as plt 
import os 
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math


class Logging: 
    def __init__(self, plot_dir: str): 
        self.plot_dir = plot_dir 
        os.makedirs(self.plot_dir, exist_ok=True) 
        self.log = dict() 

    def append(self, key: str, value): 
        if key not in self.log: 
            self.log[key] = [] 
        self.log[key].append(value) 
    
    def plot(self, x_axis_key):
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(os.path.join(self.plot_dir, "log.csv"), index=False)
        
        keys_to_plot = [k for k in self.log.keys() if k != x_axis_key]
        num_metrics = len(keys_to_plot)
        cols = 2
        rows = math.ceil(num_metrics / cols)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=False,
            vertical_spacing=0.1/rows,  
            horizontal_spacing=0.1/cols, 
            subplot_titles=keys_to_plot
        )

        x_axis = self.log[x_axis_key]
        for idx, key in enumerate(keys_to_plot):
            row = idx // cols + 1
            col = idx % cols + 1
            y_axis = self.log[key]

            fig.add_trace(
                go.Scatter(x=x_axis, y=y_axis, mode='lines', name=key, showlegend=False),
                row=row, col=col
            )

        for col in range(1, cols + 1):
            fig.update_xaxes(title_text=x_axis_key, row=rows, col=col)

        subplot_size = 300
        fig_width = (subplot_size * cols) + (60 * cols) 
        fig_height = (subplot_size * rows) + (60 * rows) 

        fig.update_layout(
            height=fig_height,
            width=fig_width,
            showlegend=False,
            title_text="Plots",
            title_x=0.5,
            margin=dict(t=60, b=60, l=60, r=60) # Maintain the original margins
)
        fig.write_html(os.path.join(self.plot_dir, "Plots.html"))


            

