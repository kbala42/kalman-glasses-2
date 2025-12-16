import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Case 9: Kalman's Glasses", page_icon="👓")

st.title("👓 Case 9: Kalman's Glasses – Seeing the Noise")
st.markdown("""
**Case:** Our drone's GPS sensor is malfunctioning and sending very noisy data. 
If we believe this data, the drone will crash. If we don't believe it, it will get lost.
**Task:** Extract the "true route" from noisy data by adjusting the Kalman Filter parameters ($Q$ and $R$).
""")

# ----------------------------------------
# 1. SETTINGS (Detective Panel)
# ----------------------------------------
st.sidebar.header("🔧 Filter Settings")

# R: Measurement Noise (Sensor Noise)
R_val = st.sidebar.slider("Sensor Noise (R) - 'Lie Rate'",
                          0.1, 100.0, 20.0, help="If it's too high, we can't trust the sensor.")

# Q: Process Noise (Model Noise)
Q_val = st.sidebar.slider("Model Uncertainty (Q) - 'Wind/Surprise'",
                          0.01, 5.0, 0.1, help="If it's high, we don't trust the model (we say the system is too variable).")

st.sidebar.markdown("---")
dt = 0.1
t_max = 20.0
steps = int(t_max / dt)

# ----------------------------------------
# 2. SIMULATION (Real World + Noise)
# ----------------------------------------
# Ground Truth
t = np.linspace(0, t_max, steps)
# Make the drone move in a sine wave pattern + add a bit of randomness.
x_true = 10 * np.sin(0.5 * t)
v_true = 10 * 0.5 * np.cos(0.5 * t)

# Sensor Data (Noise Measurement)
# We are adding Random Noise to the real data.
np.random.seed(42)
noise = np.random.normal(0, np.sqrt(R_val), steps)
z_meas = x_true + noise  # GPS only measures location

# ----------------------------------------
# 3. KALMAN FILTER MOTOR
# ----------------------------------------
# Initial Estimates
x_est = np.array([[0.0], [0.0]])  # [Position, Velocity]
P_est = np.eye(2) * 1000  # We are very uncertain at the beginning (Covariance is large)

# Matrices (Physics, as we remember from Case 8)
# x = x + v*dt
# v = v (constant velocity model assumption)
A = np.array([[1, dt],
              [0, 1]])

# Sensor Matrix (We are only measuring position: 1*x + 0*v)
H = np.array([[1, 0]])

# Noise Matrices
Q = np.array([[0.1, 0], [0, 0.1]]) * Q_val  # Process Noise Matrix
R = np.array([[R_val]])  # Measurement Noise Matrix

# Records
history_est = []
history_P = []
K_gain_history = []

for i in range(steps):
    # --- A. PREDICT (Time Update) ---
    x_pred = A @ x_est
    # DÜZELTME: AT yerine A.T kullanılmalı (Transpoz)
    P_pred = A @ P_est @ A.T + Q
    
    # --- B. UPDATE (Measurement Update) ---
    z = z_meas[i]  # Current GPS reading
    
    # Innovation (The difference between expectation and reality)
    y = z - (H @ x_pred)
    
    # Kalman Earnings (K)
    # DÜZELTME: HT yerine H.T kullanılmalı (Transpoz)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Final Prediction
    x_est = x_pred + K @ y
    P_est = (np.eye(2) - K @ H) @ P_pred
    
    # Save
    history_est.append(x_est[0, 0])
    K_gain_history.append(K[0, 0])

# ----------------------------------------
# 4. VISUALIZATION (Crime Scene)
# ----------------------------------------
st.subheader("📊 Drone Route: Actual vs Measured vs Estimated")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, x_true, 'k--', label='Actual Route (Unknown)', alpha=0.5)
ax.scatter(t, z_meas, c='r', s=10, label='GPS Measurement (Noisy)', alpha=0.3)
ax.plot(t, history_est, 'g-', linewidth=2, label='Kalman Filter (Estimation)')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Location (m)")
ax.set_title(f"Kalman Performance (R={R_val}, Q={Q_val})")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Detailed Analysis
col1, col2 = st.columns(2)
with col1:
    st.info("""
    **How to Read a Chart?**
    * **Black Line:** The actual location of the drone (We wouldn't normally know this).
    * **Red Dots:** These are lies the sensor is telling us.
    * **Green Line:** Sherlock's (Kalman) prediction.
    """)
with col2:
    error_raw = np.mean(np.abs(z_meas - x_true))
    error_kalman = np.mean(np.abs(history_est - x_true))
    
    st.metric("Sensor Error (Average)", f"{error_raw:.2f} m")
    st.metric("Kalman Error (Average)", f"{error_kalman:.2f} m",
              delta=f"{-(error_raw-error_kalman):.2f} m (Recovery)")

st.subheader("🧠 Sherlock's Confidence Graph (Kalman Gain)")
st.line_chart(pd.DataFrame(K_gain_history, columns=["Kalman Gain (K)"]))
st.caption("""
* **If K is high:** "I trust the sensor."
* **If K is low:** "The sensor is too noisy; I'm relying on my own physics model (memory)."
""")