import argparse

import plotly
import plotly.graph_objects as go
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('func', choices=['spim', 'cosine', 'norm_ratio', 'l2'])
args = parser.parse_args()

dataset = pd.read_csv(args.file)
func = args.func
last_ckpt = 585_000
dataset = dataset[
	(dataset.func == func) & (dataset.checkpoint <= last_ckpt)
].sort_values(by=['checkpoint', 'layer_idx'])

every = 1
ckpts = dataset.checkpoint.unique().tolist()[every-1::every]

# make list of continents
terms = list('ISTFC')
# make figure
fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}

# fill in most of layout
fig_dict["layout"]["xaxis"] = {"range": [-1,7], "title": "layer"}
lo = min((dataset[f'mean {t}'] - dataset[f'std {t}']).min() for t in 'ISTFC')
hi = max((dataset[f'mean {t}'] + dataset[f'std {t}']).max() for t in 'ISTFC')
extra = (hi - lo) * 5 / 100
fig_dict["layout"]["yaxis"] = {"range": [lo - extra, hi + extra], "title": func}
# fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 10, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 5,
                                                                    "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Checkpoint: ",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 5, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}

ckpt = dataset.checkpoint.min()
# make data
for term in terms:
    key = 'mean ' + term
    data_dict = {
        "x": dataset[dataset.checkpoint == ckpt]['layer_idx'].to_list(),
        "y": dataset[dataset.checkpoint == ckpt]['mean ' + term].to_list(),
        "error_y": {
            "type": "data",
            "array": dataset[dataset.checkpoint == ckpt]['std ' + term].to_list(),
        },
        "mode": "lines",
        "name": term
    }
    fig_dict["data"].append(data_dict)

# make frames
for ckpt in ckpts:
    frame = {"data": [], "name": str(ckpt)}
    for term in terms:
        data_dict = {
            "x": dataset[dataset.checkpoint == ckpt]['layer_idx'].to_list(),
            "y": dataset[dataset.checkpoint == ckpt]['mean ' + term].to_list(),
            "error_y": {
                "type": "data",
                "array": dataset[dataset.checkpoint == ckpt]['std ' + term].to_list(),
            },
            "mode": "lines",
            "name": term,
        }
        frame["data"].append(data_dict)

    fig_dict["frames"].append(frame)
    slider_step = {"args": [
        [ckpt],
        {"frame": {"duration": 5, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 5}}
    ],
        "label": ckpt,
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)


fig_dict["layout"]["sliders"] = [sliders_dict]

print('make fig')
fig = go.Figure(fig_dict) 

print('show fig')
fig.show()
