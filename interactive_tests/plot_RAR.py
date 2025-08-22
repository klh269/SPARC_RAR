#!/usr/bin/env python
# Interactive RAR plot using Plotly.
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Magic to make sure we can import utils_analysis.Vobs_fits from the parent directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils_analysis.Vobs_fits import MOND_vsq
from utils_analysis.get_SPARC import get_SPARC_data
from utils_analysis.MOND_IFs import n_family, delta_family, gamma_family, alpha_family, rar_if, simple_if, standard_if

# --- Create figure with 1 RAR plot and 2 RC + Acc subplots ---
def create_subplots():
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"rowspan": 2}, {}],  # Row 1: RAR in col 1 (rowspan), Acc curves in col 2
            [None, {}]             # Row 2: empty under RAR, RC in col 2
        ],
        subplot_titles=("RAR", "Rotation Curves", "Acceleration Curves"),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    return fig

def superscript_number(n):
    super_digits = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(n).translate(super_digits)

# --- Plot with galaxy selection for RAR, RC and Acceleration curves ---
def plot_per_galaxy():
    fig = create_subplots()

    # Get SPARC data and flatten into background arrays
    sparc_data, galaxies, galaxy_count = get_SPARC_data()
    all_g_obs, all_g_bar, all_galaxy_names = [], [], []
    for gal, vals in sparc_data.items():
        all_g_obs.extend(vals['g_obs'])
        all_g_bar.extend(vals['g_bar'])
        all_galaxy_names.extend([gal] * len(vals['g_bar']))  # repeat galaxy name for each point

    # --- Subplot (1,1): RAR ---
    # Background points
    fig.add_trace(go.Scatter(
        x=all_g_bar,
        y=all_g_obs,
        mode='markers',
        marker=dict(color='lightgrey', size=5),
        customdata=all_galaxy_names,
        hovertemplate="Galaxy: %{customdata}<br>g_bar: %{x:.2e}<br>g_obs: %{y:.2e}<extra></extra>",
        name="All galaxies",
        legendgroup="rar",
        showlegend=True
    ), row=1, col=1)

    # y=x reference line + 3 special IF lines
    x_line = np.logspace(-13, -7.5, 100)
    fig.add_trace(go.Scatter( x=x_line, y=x_line, mode='lines', name='y = x',
        line=dict(color='black', width=2, dash='dash') ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=simple_if(x_line), mode='lines', name='Simple IF',
        line=dict(color='red', width=2) ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=standard_if(x_line), mode='lines', name='Standard IF',
        line=dict(color='green', width=2) ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=rar_if(x_line), mode='lines', name='RAR IF',
        line=dict(color='magenta', width=2) ), row=1, col=1)

    # Add one highlighted trace per galaxy
    for gal in galaxies:
        vals = sparc_data[gal]
        # --- Subplot (1,1): Highlighted points on RAR ---
        fig.add_trace(go.Scatter(
            x=vals['g_bar'], y=vals['g_obs'],
            mode='markers', marker=dict(size=8, color='blue'),
            name=f"{gal} RAR",
            visible=(gal == galaxies[0]),
            legendgroup="rar",
            showlegend=False
        ), row=1, col=1)

        # --- Subplot (1,2): Rotation Curves ---
        # Rotation curve: Vobs
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['Vobs'],
            error_y=dict( type='data', array=vals['errV'], visible=True,
                color='black', thickness=1.5, width=4 ),
            mode='markers+lines',
            line=dict(color='black'),
            name=f"{gal} Vobs/g_obs",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=True
        ), row=1, col=2)
        # Rotation curve: Vbar
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['Vbar'],
            mode='lines+markers', line=dict(color='firebrick'),
            name=f"{gal} Vbar/g_bar",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=True
        ), row=1, col=2)

        # Simple IF for Vobs
        fig.add_trace(go.Scatter(
            x=vals['r'],
            y=np.sqrt(MOND_vsq(vals['r'], vals['Vbar']**2)),
            mode='lines',
            line=dict(color='red', width=2),
            name=f"{gal} Simple IF",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=False
        ), row=1, col=2)
        # Standard IF for Vobs
        fig.add_trace(go.Scatter(
            x=vals['r'],
            y=np.sqrt(standard_if(vals['g_bar']) * vals['r'] * 3.086e19) / 1e3,  # Convert to km/s
            mode='lines',
            line=dict(color='green', width=2),
            name=f"{gal} Standard IF",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=False
        ), row=1, col=2)

        # --- Subplot (1,2): 'Acceleration Curves' ---
        # Acceleration curve: g_obs
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['g_obs'],
            mode='lines+markers', line=dict(color='black'),
            name=f"{gal} g_obs",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)
        # Acceleration curve: g_bar
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['g_bar'],
            mode='lines+markers', line=dict(color='firebrick'),
            name=f"{gal} g_bar",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)

        # Simple IF for g_obs
        fig.add_trace(go.Scatter(
            x=vals['r'],
            y=simple_if(vals['g_bar']),
            mode='lines',
            line=dict(color='red', width=2),
            name=f"{gal} Simple IF",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)
        # Standard IF for g_obs
        fig.add_trace(go.Scatter(
            x=vals['r'],
            y=standard_if(vals['g_bar']),
            mode='lines',
            line=dict(color='green', width=2),
            name=f"{gal} Standard IF",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)

    # --- Build interactive dropdown button ---
    n_fixed_traces   = 5    # 5 fixed traces: background + y=x + 3 special IFs
    n_traces_per_gal = 9    # 9 traces per galaxy: RAR + 4 x RC + 4 x Acc curves
    buttons = []
    vis_fixed = [True] * n_fixed_traces

    for g_idx, gal in enumerate(galaxies):
        vis_per_gal = []
        for gg in range(galaxy_count): vis_per_gal += [gg == g_idx] * n_traces_per_gal
        vis_list = vis_fixed + vis_per_gal  # Combine fixed traces with galaxy-specific traces
        buttons.append(dict( label=gal, method='update', args=[{'visible': vis_list}] ))

    # --- Layout ---
    exponents = np.arange(-13, -7)
    tickvals = [10.0**e for e in exponents]
    ticktext = [f"10{superscript_number(e)}" for e in exponents]

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                        x=1.05, y=1.1, xanchor='left', yanchor='top')],
        title=f"RAR & Rotation Curves of SPARC galaxies",
        height=700, width=900,
        xaxis=dict(title=f"g_bar (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title=f"g_obs (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),
        xaxis2=dict(title=" "),                                             # RC X-axis
        yaxis2=dict(title=f"Velocities (kms{superscript_number(-1)})"),     # RC Y-axis
        xaxis3=dict(title="Radius (kpc)"),                                  # Acc X-axis
        yaxis3=dict(title=f"g (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),                      # Acc Y-axis
        legend=dict(x=1.05, y=1, xanchor='left')
    )

    fig.write_html("/mnt/users/koe/SPARC_RAR/plots/interactive_IF_plots/galaxy_selection.html")

# --- Plot same galaxy selection but for unhooked RAR (from rm_hooks.py) ---
def plot_unhooked_RAR():
    fig = create_subplots()

    # Get SPARC data and flatten into background arrays
    sparc_data, _, _ = get_SPARC_data()
    galaxies = [
        "D564-8", "UGC00731", "D631-7", "UGC04278", "DDO154",
        "UGC05414", "DDO168", "UGC05764", "ESO116-G012", "UGC05986",
        "F574-1", "UGC06667", "IC2574", "UGC06917", "KK98-251", "UGC07089",
        "NGC0055", "UGC07151", "NGC0100", "UGC07399", "NGC2403", "UGC07603",
        "NGC3109", "UGC08837", "NGC4010", "UGCA442" ]   # 24 'hooked' galaxies from https://arxiv.org/pdf/2307.09507
    galaxy_count = len(galaxies)
    
    all_g_obs, all_g_bar, all_galaxy_names = [], [], []
    for gal, vals in sparc_data.items():
        all_g_obs.extend(vals['g_obs'])
        all_g_bar.extend(vals['g_bar'])
        all_galaxy_names.extend([gal] * len(vals['g_bar']))  # repeat galaxy name for each point

    # --- Subplot (1,1): RAR ---
    # Background points
    fig.add_trace(go.Scatter(
        x=all_g_bar,
        y=all_g_obs,
        mode='markers',
        marker=dict(color='lightgrey', size=5),
        customdata=all_galaxy_names,
        hovertemplate="Galaxy: %{customdata}<br>g_bar: %{x:.2e}<br>g_obs: %{y:.2e}<extra></extra>",
        name="All galaxies",
        legendgroup="rar",
        showlegend=True
    ), row=1, col=1)

    # y=x reference line + 3 special IF lines
    x_line = np.logspace(-13, -7.5, 100)
    fig.add_trace(go.Scatter( x=x_line, y=x_line, mode='lines', name='y = x',
        line=dict(color='black', width=2, dash='dash') ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=simple_if(x_line), mode='lines', name='Simple IF',
        line=dict(color='red', width=2) ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=standard_if(x_line), mode='lines', name='Standard IF',
        line=dict(color='green', width=2) ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=rar_if(x_line), mode='lines', name='RAR IF',
        line=dict(color='magenta', width=2) ), row=1, col=1)

    # Add one highlighted trace per galaxy
    for gal in galaxies:
        vals = sparc_data[gal]
        # --- Subplot (1,1): Highlighted points on RAR ---
        fig.add_trace(go.Scatter(
            x=vals['g_bar'], y=vals['g_obs'],
            mode='markers', marker=dict(size=8, color='black'),
            name=f"{gal} RAR",
            visible=(gal == galaxies[0]),
            legendgroup="rar",
            showlegend=True
        ), row=1, col=1)
        # Unhooked RAR points
        unhooked_RAR = np.load("/mnt/users/koe/SPARC_RAR/unhooked_RAR.npy", allow_pickle=True).item()
        fig.add_trace(go.Scatter(
            x=unhooked_RAR[gal]['g_bar'], y=unhooked_RAR[gal]['g_obs'],
            mode='markers', marker=dict(size=8, color='red', symbol='diamond'),
            name=f"{gal} Unhooked RAR",
            visible=(gal == galaxies[0]),
            legendgroup="rar",
            showlegend=True
        ), row=1, col=1)

        # --- Subplot (1,2): Rotation Curves ---
        # Rotation curve: Vobs
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['Vobs'],
            error_y=dict( type='data', array=vals['errV'], visible=True,
                color='black', thickness=1.5, width=4 ),
            mode='markers+lines',
            line=dict(color='black'),
            name=f"{gal} Vobs/g_obs",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=True
        ), row=1, col=2)
        # Rotation curve: Vbar
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['Vbar'],
            mode='lines+markers', line=dict(color='firebrick'),
            name=f"{gal} Vbar/g_bar",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=True
        ), row=1, col=2)

        # Unhooked RAR velocities
        fig.add_trace(go.Scatter(
            x=vals['r'], y=unhooked_RAR[gal]['Vobs'],
            mode='lines+markers', marker=dict(size=8, color='blue', symbol='diamond'),
            name=f"{gal} Unhooked Vobs/g_obs",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=True
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=vals['r'], y=unhooked_RAR[gal]['Vbar'],
            mode='lines+markers', marker=dict(size=8, color='red', symbol='diamond'),
            name=f"{gal} Unhooked Vbar/g_bar",
            visible=(gal == galaxies[0]),
            legendgroup="rc",
            showlegend=True
        ), row=1, col=2)

        # # Simple IF for Vobs
        # fig.add_trace(go.Scatter(
        #     x=vals['r'],
        #     y=np.sqrt(MOND_vsq(vals['r'], vals['Vbar']**2)),
        #     mode='lines',
        #     line=dict(color='red', width=2),
        #     name=f"{gal} Simple IF",
        #     visible=(gal == galaxies[0]),
        #     legendgroup="rc",
        #     showlegend=False
        # ), row=1, col=2)
        # # Standard IF for Vobs
        # fig.add_trace(go.Scatter(
        #     x=vals['r'],
        #     y=np.sqrt(standard_if(vals['g_bar']) * vals['r'] * 3.086e19) / 1e3,  # Convert to km/s
        #     mode='lines',
        #     line=dict(color='green', width=2),
        #     name=f"{gal} Standard IF",
        #     visible=(gal == galaxies[0]),
        #     legendgroup="rc",
        #     showlegend=False
        # ), row=1, col=2)

        # --- Subplot (1,2): 'Acceleration Curves' ---
        # Acceleration curve: g_obs
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['g_obs'],
            mode='lines+markers', line=dict(color='black'),
            name=f"{gal} g_obs",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)
        # Acceleration curve: g_bar
        fig.add_trace(go.Scatter(
            x=vals['r'], y=vals['g_bar'],
            mode='lines+markers', line=dict(color='firebrick'),
            name=f"{gal} g_bar",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)

        # Unhooked RAR accelerations
        fig.add_trace(go.Scatter(
            x=vals['r'], y=unhooked_RAR[gal]['g_obs'],
            mode='lines+markers', marker=dict(size=8, color='blue', symbol='diamond'),
            name=f"{gal} Unhooked g_obs",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=vals['r'], y=unhooked_RAR[gal]['g_bar'],
            mode='lines+markers', marker=dict(size=8, color='red', symbol='diamond'),
            name=f"{gal} Unhooked g_bar",
            visible=(gal == galaxies[0]),
            legendgroup="acc",
            showlegend=False
        ), row=2, col=2)

        # # Simple IF for g_obs
        # fig.add_trace(go.Scatter(
        #     x=vals['r'],
        #     y=simple_if(vals['g_bar']),
        #     mode='lines',
        #     line=dict(color='red', width=2),
        #     name=f"{gal} Simple IF",
        #     visible=(gal == galaxies[0]),
        #     legendgroup="acc",
        #     showlegend=False
        # ), row=2, col=2)
        # # Standard IF for g_obs
        # fig.add_trace(go.Scatter(
        #     x=vals['r'],
        #     y=standard_if(vals['g_bar']),
        #     mode='lines',
        #     line=dict(color='green', width=2),
        #     name=f"{gal} Standard IF",
        #     visible=(gal == galaxies[0]),
        #     legendgroup="acc",
        #     showlegend=False
        # ), row=2, col=2)

    # --- Build interactive dropdown button ---
    n_fixed_traces   = 5    # 5 fixed traces: background + y=x + 3 special IFs
    n_traces_per_gal = 10   # 10 traces per galaxy: 2 x RAR + 4 x RC + 4 x Acc curves
    buttons = []
    vis_fixed = [True] * n_fixed_traces

    for g_idx, gal in enumerate(galaxies):
        vis_per_gal = []
        for gg in range(galaxy_count): vis_per_gal += [gg == g_idx] * n_traces_per_gal
        vis_list = vis_fixed + vis_per_gal  # Combine fixed traces with galaxy-specific traces
        buttons.append(dict( label=gal, method='update', args=[{'visible': vis_list}] ))

    # --- Layout ---
    exponents = np.arange(-13, -7)
    tickvals = [10.0**e for e in exponents]
    ticktext = [f"10{superscript_number(e)}" for e in exponents]

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                        x=1.05, y=1.1, xanchor='left', yanchor='top')],
        title=f"RAR & Rotation Curves of SPARC galaxies",
        height=700, width=900,
        xaxis=dict(title=f"g_bar (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title=f"g_obs (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),
        xaxis2=dict(title=" "),                                             # RC X-axis
        yaxis2=dict(title=f"Velocities (kms{superscript_number(-1)})"),     # RC Y-axis
        xaxis3=dict(title="Radius (kpc)"),                                  # Acc X-axis
        yaxis3=dict(title=f"g (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),                      # Acc Y-axis
        legend=dict(x=1.05, y=1, xanchor='left')
    )

    fig.write_html("/mnt/users/koe/SPARC_RAR/plots/unhooked_RAR.html")

# --- Plot RAR families with interactive slider ---
def plot_RAR_families(family:str="delta", gal:str="UGC06787"):
    if family not in ("n", "delta", "gamma", "alpha"):
        raise ValueError(f"Invalid family '{family}'. Must be one of: 'n', 'delta', 'gamma' or 'alpha'.")

    fig = create_subplots()

    # Get SPARC data and flatten into background arrays
    sparc_data, _, _ = get_SPARC_data()

    # Check if the specified galaxy exists in SPARC
    if gal not in sparc_data:
        raise ValueError(f"Galaxy '{gal}' not found in SPARC data.")
    
    all_g_obs, all_g_bar, all_galaxy_names = [], [], []
    for g, vals in sparc_data.items():
        all_g_obs.extend(vals['g_obs'])
        all_g_bar.extend(vals['g_bar'])
        all_galaxy_names.extend([g] * len(vals['g_bar']))   # repeat galaxy name for each point

    # --- Subplot (1,1): RAR ---
    # Background points
    fig.add_trace(go.Scatter(
        x=all_g_bar,
        y=all_g_obs,
        mode='markers',
        marker=dict(color='lightgrey', size=5),
        customdata=all_galaxy_names,
        hovertemplate="Galaxy: %{customdata}<br>g_bar: %{x:.2e}<br>g_obs: %{y:.2e}<extra></extra>",
        name="All galaxies",
        legendgroup="rar",
        showlegend=True
    ), row=1, col=1)

    # y=x reference line + 3 special IF lines
    x_line = np.logspace(-13, -7.5, 100)
    fig.add_trace(go.Scatter( x=x_line, y=x_line, mode='lines', name='y = x',
        line=dict(color='black', width=2, dash='dash') ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=simple_if(x_line), mode='lines', name='Simple IF',
        line=dict(color='orange', width=1.5) ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=standard_if(x_line), mode='lines', name='Standard IF',
        line=dict(color='green', width=1.5) ), row=1, col=1)
    fig.add_trace(go.Scatter( x=x_line, y=rar_if(x_line), mode='lines', name='RAR IF',
        line=dict(color='magenta', width=1.5) ), row=1, col=1)
    
    vals = sparc_data[gal]  # Example galaxy w/ features (default: UGC 6787)

    # Highlighted trace for selected galaxy (gal)
    fig.add_trace(go.Scatter(
        x=vals['g_bar'], y=vals['g_obs'],
        mode='markers', marker=dict(size=8, color='blue'),
        name=f"{gal} RAR",
        visible=True,
        legendgroup="rar",
        showlegend=True
    ), row=1, col=1)

    # --- Subplot (1,2): Rotation Curves ---
    # Rotation curve: Vobs
    fig.add_trace(go.Scatter(
        x=vals['r'],
        y=vals['Vobs'],
        error_y=dict( type='data', array=vals['errV'], visible=True,
            color='black', thickness=1.5, width=4 ),
        mode='markers+lines',
        line=dict(color='black'),
        name=f"{gal} Vobs/g_obs",
        visible=True,
        legendgroup="rc",
        showlegend=True
    ), row=1, col=2)
    # Rotation curve: Vbar
    fig.add_trace(go.Scatter(
        x=vals['r'], y=vals['Vbar'],
        mode='lines+markers', line=dict(color='firebrick'),
        name=f"{gal} Vbar/g_bar",
        visible=True,
        legendgroup="rc",
        showlegend=True
    ), row=1, col=2)

    # --- Subplot (1,2): 'Acceleration Curves' ---
    # Acceleration curve: g_obs
    fig.add_trace(go.Scatter(
        x=vals['r'], y=vals['g_obs'],
        mode='lines+markers', line=dict(color='black'),
        name=f"{gal} g_obs",
        visible=True,
        legendgroup="acc",
        showlegend=False
    ), row=2, col=2)
    # Acceleration curve: g_bar
    fig.add_trace(go.Scatter(
        x=vals['r'], y=vals['g_bar'],
        mode='lines+markers', line=dict(color='firebrick'),
        name=f"{gal} g_bar",
        visible=True,
        legendgroup="acc",
        showlegend=False
    ), row=2, col=2)

    n_fixed_traces = 10     # 10 fixed traces: background + y=x + 3 special IFs + highlighted RAR + 2 RCs + 2 Acc curves

    # --- IF family traces (slider-controlled) ---
    param_values = np.linspace(0.5, 5.0, int((5.0 - 0.5) / 0.1) + 1)
    n_values = len(param_values)

    if family == "n": 
        get_IF_family = n_family
        symbol = "n"
    elif family == "delta":
        get_IF_family = delta_family
        symbol = "δ"
    elif family == "gamma":
        get_IF_family = gamma_family
        symbol = "γ"
    elif family == "alpha":
        get_IF_family = alpha_family
        symbol = "α"

    for j, p_val in enumerate(param_values):
        # --- Subplot (1,1): RAR ---
        fig.add_trace(go.Scatter(
            x=x_line,
            y=get_IF_family(p_val, x_line),
            mode='lines',
            line=dict(color='red', width=2),
            name=f"{symbol} = {p_val:.1f}",
            visible=(j==0),
            legendgroup="if_family",
            showlegend=True
        ), row=1, col=1)

        # --- Subplot (1,2): RC ---
        Vobs_family = ( get_IF_family(p_val, vals['g_bar']) * vals['r'] * 3.086e19 )**0.5 / 1e3    # Convert to km/s
        fig.add_trace(go.Scatter(
            x=vals['r'],
            y=Vobs_family,
            mode='lines',
            line=dict(color='red', width=2),
            name=f"{symbol} = {p_val:.1f}",
            visible=(j==0),
            legendgroup="if_family",
            showlegend=False
        ), row=1, col=2)

        # --- Subplot (2,2): Acceleration ---
        fig.add_trace(go.Scatter(
            x=vals['r'],
            y=get_IF_family(p_val, vals['g_bar']),
            mode='lines',
            line=dict(color='red', width=2),
            name=f"{symbol} = {p_val:.1f}",
            visible=(j==0),
            legendgroup="if_family",
            showlegend=False
        ), row=2, col=2)

    # --- Interactive slider for selecting delta values ---
    steps = []
    for i, delta_val in enumerate(param_values):
        vis_per_val = []
        for vv in range(n_values): vis_per_val += [vv == i] * 3         # 3 traces per delta value: RAR + RC + Acc
        step = dict(
            method="update",
            args=[{"visible": [True] * n_fixed_traces + vis_per_val}],  # 10 fixed traces + delta traces
            label=f"{symbol} = {delta_val:.1f}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Current value: "},
        pad={"t": 50},
        steps=steps
    )]

    # Make the first delta trace (δ = 0.5) visible by default
    for idx, trace in enumerate(fig.data[n_fixed_traces:]): trace.visible = (idx in [0, 1, 2])

    # --- Layout ---
    exponents = np.arange(-13, -7)
    tickvals = [10.0**e for e in exponents]
    ticktext = [f"10{superscript_number(e)}" for e in exponents]

    fig.update_layout(
        sliders=sliders,
        title=f"RAR of {gal} ({symbol} family)",
        height=700, width=900,
        xaxis=dict(title=f"g_bar (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title=f"g_obs (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),
        xaxis2=dict(title=" "),                                             # RC X-axis
        yaxis2=dict(title=f"Velocities (kms{superscript_number(-1)})"),     # RC Y-axis
        xaxis3=dict(title="Radius (kpc)"),                                  # Acc X-axis
        yaxis3=dict(title=f"g (ms{superscript_number(-2)})", type='log',
                tickvals=tickvals, ticktext=ticktext),                      # Acc Y-axis
        legend=dict(x=1.05, y=1, xanchor='left')
    )

    # fig.write_html(f"/mnt/users/koe/SPARC_RAR/plots/interactive_IF_plots/{family}_family_RAR.html")
    fig.write_html(f"/mnt/users/koe/SPARC_RAR/plots/interactive_IF_plots/{family}_family/{gal}.html")


if __name__ == "__main__":
    plot_unhooked_RAR()

    # SPARC_galaxies = get_SPARC_data()[1]

    # for family in ("n", "delta", "gamma", "alpha"):
    #     print(f"Plotting {family} family for all SPARC galaxies...")
    #     for gal in SPARC_galaxies:
    #         try:
    #             plot_RAR_families(family, gal)
    #         except ValueError as e:
    #             print(f"Skipping {gal}: {e}")

# TO DO:
#   Add IFs to RC and Acc plots. Maybe add some sort of residuals too?
