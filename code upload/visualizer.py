# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys

# Add the project root to sys path to import generator
sys.path.insert(0, str(Path(__file__).parent))

from generator.topology import CampusTopology

DATA_DIR = Path(__file__).parent / 'data' / 'raw'

st.set_page_config(page_title="Telemetry Visualizer", layout="wide")

@st.cache_data
def load_data():
    ports_df = pd.read_parquet(DATA_DIR / 'ports_batch_0.parquet')
    devices_df = pd.read_parquet(DATA_DIR / 'devices_batch_0.parquet')
    return ports_df, devices_df

@st.cache_resource
def load_topology():
    return CampusTopology()

st.title("Campus Network Telemetry Dashboard")

try:
    ports_df, devices_df = load_data()
    topology = load_topology()
except Exception as e:
    st.error(f"Error loading data. Have you run the generator yet? {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Network Map", "Device Telemetry", "Port Telemetry (Storms)"])

with tab1:
    st.header("Campus Network Topology")
    st.markdown("Interactive map of the core, distribution, and access switches.")
    # Draw graph using plotly
    G = topology.graph
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
        
    node_x = []
    node_y = []
    text = []
    colors = []
    sizes = []
    symbols = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if "Core" in node:
            colors.append("red")
            text.append(f"🏢 {node}")
            sizes.append(24)
            symbols.append("square")
        elif "Dist" in node:
            colors.append("orange")
            text.append(f"🔀 {node}")
            sizes.append(18)
            symbols.append("diamond")
        else:
            colors.append("lightblue")
            text.append(f"🖥️ {node}")
            sizes.append(14)
            symbols.append("circle")
            
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=colors,
            size=sizes,
            symbol=symbols,
            line_width=1,
            line_color="black"))
            
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600)
                )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Device Telemetry")
    switch_list = sorted(devices_df['switch_id'].unique())
    selected_switch = st.selectbox("Select Switch", switch_list)
    
    switch_data = devices_df[devices_df['switch_id'] == selected_switch].copy()
    switch_data = switch_data.sort_values('timestamp')
    
    fig = px.line(switch_data, x='timestamp', y=['cpu_utilization', 'memory_utilization'], 
                  title=f"CPU and Memory Load for {selected_switch}")
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.line(switch_data, x='timestamp', y='poe_usage_watts', 
                  title=f"Power over Ethernet (PoE) Draw for {selected_switch}")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Port Telemetry & Storm Tracking")
    
    # Filter ports for the selected switch to make it easier to find
    st.markdown(f"Showing ports for **{selected_switch}**")
    switch_ports = ports_df[ports_df['switch_id'] == selected_switch]['port_id'].unique()
    
    if len(switch_ports) > 0:
        selected_port = st.selectbox("Select Port", sorted(switch_ports))
        
        port_data = ports_df[ports_df['port_id'] == selected_port].copy()
        port_data = port_data.sort_values('timestamp')
        
        # Highlight storms (safely handle NaNs from parquet)
        storm_mask = port_data['is_storm_affected'].fillna(False).astype(bool)
        storm_data = port_data[storm_mask]
        
        fig = px.line(port_data, x='timestamp', y=['rx_utilization', 'tx_utilization'],
                      title=f"Traffic Utilization for {selected_port}")
        if not storm_data.empty:
            fig.add_scatter(x=storm_data['timestamp'], y=storm_data['rx_utilization'], 
                            mode='markers', marker=dict(color='red', size=8), name="Storm Detected")
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.line(port_data, x='timestamp', y='queue_drops',
                      title=f"Queue Drops for {selected_port}")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning(f"No ports found for {selected_switch}.")
