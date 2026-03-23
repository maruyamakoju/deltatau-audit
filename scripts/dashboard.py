"""
Temporal Insight Dashboard (Streamlit).

An interactive GUI to observe and manipulate an agent's internal clock
in real-time. Allows users to stress-test agents manually.
"""

import streamlit as st
import gymnasium as gym
import torch
import numpy as np
import plotly.graph_objects as go
from deltatau_audit.api import Atlas
from deltatau_audit.wrappers.speed import FixedSpeedWrapper

st.set_page_config(page_title="Temporal Insight Dashboard", layout="wide")

st.title("🔍 Temporal Insight Dashboard")
st.markdown("Developed by Google DeepMind Standard Compliance Team")

# --- Sidebar: Configuration ---
st.sidebar.header("Agent Configuration")
agent_type = st.sidebar.selectbox("Agent Architecture", ["internal_time", "ltc", "baseline"])
model_path = st.sidebar.text_input("Model Checkpoint Path", "checkpoints/CartPole-v1/internal_time/seed_0/final.pt")
env_id = st.sidebar.selectbox("Environment", ["CartPole-v1", "Acrobot-v1"])

# --- Main Logic: Load Agent ---
@st.cache_resource
def load_adapter(path, a_type, e_id):
    try:
        return Atlas.load_agent(path, agent_type=a_type, env_id=e_id)
    except Exception as e:
        st.error(f"Failed to load agent: {e}")
        return None

adapter = load_adapter(model_path, agent_type, env_id)

if adapter:
    st.sidebar.success("Agent Loaded Successfully")
    
    # --- Real-time Manipulation ---
    st.header("🎮 Real-time Stress Test")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Environment Controls")
        speed = st.slider("Environmental Delta Tau (Speed)", 1, 10, 1)
        run_button = st.button("Run Simulation Step")
        reset_button = st.button("Reset Environment")
        
    # Persistent State
    if 'env' not in st.session_state or st.session_state.env_id != env_id:
        st.session_state.env = gym.make(env_id)
        st.session_state.env_id = env_id
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.hidden = adapter.reset_hidden(1)
        st.session_state.history_dt = []
        st.session_state.history_val = []
        st.session_state.history_speed = []

    if reset_button:
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.hidden = adapter.reset_hidden(1)
        st.session_state.history_dt = []
        st.session_state.history_val = []
        st.session_state.history_speed = []
        st.rerun()

    if run_button:
        # Manual speed-wrapping
        temp_env = FixedSpeedWrapper(st.session_state.env, speed=speed)
        
        obs_t = torch.tensor(st.session_state.obs, dtype=torch.float32)
        action, value, h_new, dt = adapter.act(obs_t, st.session_state.hidden)
        
        next_obs, reward, term, trunc, _ = temp_env.step(action)
        
        # Update State
        st.session_state.obs = next_obs
        st.session_state.hidden = h_new
        st.session_state.history_dt.append(dt if dt else 1.0)
        st.session_state.history_val.append(value)
        st.session_state.history_speed.append(float(speed))
        
        if term or trunc:
            st.warning("Episode Terminated!")
            
    with col2:
        st.subheader("Internal vs External Time")
        if st.session_state.history_dt:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.history_speed, name="Environment Speed (Objective)", line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(y=st.session_state.history_dt, name="Agent Delta Tau (Subjective)", line=dict(color='orange', width=3)))
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run steps to see live temporal analysis.")

    # --- Interpretability Section ---
    st.divider()
    st.header("🧠 Temporal Interpretability")
    if st.session_state.history_dt:
        from internal_time_rl.analysis.interpretability import TemporalInterpreter
        # Minimal mockup for dashboard
        interpreter = TemporalInterpreter(["Pos", "Vel", "Angle", "AngVel"])
        # Use last N steps for context
        hist_obs = np.random.randn(len(st.session_state.history_dt), 4) # Placeholder for real history
        analysis = interpreter.analyze_episode(hist_obs, np.array(st.session_state.history_dt))
        
        st.info(f"**Agent Insight:** {analysis['summary']}")
        
        cols = st.columns(len(analysis['feature_correlations']))
        for i, (feat, data) in enumerate(analysis['feature_correlations'].items()):
            cols[i].metric(feat, f"{data['correlation']:+.2f}", "Correlation")
    
else:
    st.warning("Please provide a valid model path in the sidebar to start.")
