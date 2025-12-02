import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Beam Studio Pro", layout="wide", page_icon="🏗️")

# --- CUSTOM CSS FOR DARK ENGINEERING THEME ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007acc; color: white; border: none; }
    .stButton>button:hover { background-color: #0098ff; }
    h1, h2, h3 { color: #0098ff; font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE (MEMORY) ---
# Unlike Tkinter, Streamlit reruns the script on every click. 
# We use session_state to remember the loads.
if 'loads_list' not in st.session_state:
    st.session_state.loads_list = []

# --- LOGIC & PHYSICS (EXACTLY SAME AS YOUR TKINTER CODE) ---
def validate_inputs(load_type, L, *args):
    try:
        if L <= 0: return False, "Beam length must be positive."
        if load_type == "POINT":
            loc = args[0]
            if loc < 0 or loc > L: return False, f"Location {loc}m is outside beam (0-{L}m)."
        elif load_type in ["UDL", "UVL"]:
            s, e = args[0], args[1]
            if s < 0 or e > L: return False, f"Range {s}-{e}m is outside beam (0-{L}m)."
            if s == e: return False, "Start and End cannot be the same point."
        return True, ""
    except: return False, "Invalid numeric input."

def calculate_analysis(L, SupA, SupB, b_type, loads):
    # Determine Reactions
    m_sum = 0; f_sum = 0
    
    for l in loads:
        if l["type"] == "POINT":
            f = l["mag"]; m_sum += f*(l["loc"]-SupA); f_sum += f
        elif l["type"] == "UDL":
            length = l["end"]-l["start"]; f = l["mag"]*length
            m_sum += f*(l["start"] + length/2 - SupA); f_sum += f
        elif l["type"] == "UVL":
            w1, w2 = l["start_mag"], l["end_mag"]; length = l["end"]-l["start"]
            f1 = min(w1,w2)*length; f2 = 0.5*abs(w2-w1)*length
            c1 = l["start"] + length/2 - SupA
            c2 = l["start"] + (2/3)*length - SupA if w2 > w1 else l["start"] + (1/3)*length - SupA
            m_sum += (f1*c1) + (f2*c2); f_sum += (f1+f2)
            
    if b_type == "Cantilever": Rb = 0; Ra = f_sum; Ma = m_sum
    else: Rb = m_sum / (SupB - SupA); Ra = f_sum - Rb; Ma = 0
    
    # Generate Arrays
    x = np.linspace(0, L, 500); V = []; M = []
    for xi in x:
        v_val = 0; m_val = 0
        if xi > SupA: v_val += Ra; m_val += Ra*(xi-SupA) - Ma
        if b_type != "Cantilever" and xi > SupB: v_val += Rb; m_val += Rb*(xi-SupB)
        
        for l in loads:
            if l["type"] == "POINT" and xi > l["loc"]:
                v_val -= l["mag"]; m_val -= l["mag"]*(xi-l["loc"])
            elif l["type"] == "UDL" and xi > l["start"]:
                ov = min(xi, l["end"]) - l["start"]
                f = l["mag"] * ov
                v_val -= f; m_val -= f * (xi - (l["start"] + ov/2))
            elif l["type"] == "UVL" and xi > l["start"]:
                curr_end = min(xi, l["end"]); dx = curr_end - l["start"]
                w1 = l["start_mag"]
                w_at_x = w1 + ((l["end_mag"]-w1)*(dx/(l["end"]-l["start"])))
                f_slice = ((w1+w_at_x)/2)*dx
                cent = (dx/3)*((2*w1+w_at_x)/(w1+w_at_x))
                v_val -= f_slice; m_val -= f_slice*(xi-(l["start"]+cent))
        V.append(v_val); M.append(m_val)
        
    return x, V, M, Ra, Rb, Ma

# --- GUI LAYOUT ---
st.title("🏗️ Beam Studio Pro")
st.caption("Professional Structural Analysis Tool (Web Edition)")

col_setup, col_viz = st.columns([1, 2])

# === SIDEBAR (SETUP) ===
with col_setup:
    st.subheader("1. Structure")
    b_type = st.selectbox("Support Type", ["Simply Supported", "Cantilever"])
    L = st.number_input("Beam Length (m)", value=10.0, step=1.0)
    SupA = st.number_input("Support A (m)", value=0.0, step=0.5)
    SupB = st.number_input("Support B (m)", value=10.0, step=0.5)

    st.subheader("2. Load Manager")
    tab1, tab2, tab3 = st.tabs(["Point", "UDL", "UVL"])
    
    with tab1:
        pm = st.number_input("Mag (kN)", value=10.0, key="pm")
        pl = st.number_input("Loc (m)", value=5.0, key="pl")
        if st.button("Add Point Load"):
            valid, msg = validate_inputs("POINT", L, pl)
            if valid: st.session_state.loads_list.append({"type":"POINT", "mag":pm, "loc":pl})
            else: st.error(msg)
            
    with tab2:
        um = st.number_input("Mag (kN/m)", value=5.0, key="um")
        us = st.number_input("Start (m)", value=2.0, key="us")
        ue = st.number_input("End (m)", value=8.0, key="ue")
        if st.button("Add UDL"):
            if us > ue: us, ue = ue, us # Smart Swap
            valid, msg = validate_inputs("UDL", L, us, ue)
            if valid: st.session_state.loads_list.append({"type":"UDL", "mag":um, "start":us, "end":ue})
            else: st.error(msg)

    with tab3:
        vm1 = st.number_input("Start Mag", value=0.0, key="vm1")
        vm2 = st.number_input("End Mag", value=10.0, key="vm2")
        vs = st.number_input("Start (m)", value=2.0, key="vs")
        ve = st.number_input("End (m)", value=8.0, key="ve")
        if st.button("Add UVL"):
            if vs > ve: 
                vs, ve = ve, vs
                vm1, vm2 = vm2, vm1 # Smart Swap
            valid, msg = validate_inputs("UVL", L, vs, ve)
            if valid: st.session_state.loads_list.append({"type":"UVL", "start_mag":vm1, "end_mag":vm2, "start":vs, "end":ve})
            else: st.error(msg)

    # Load List Display
    st.markdown("---")
    st.write("**Current Loads:**")
    if st.session_state.loads_list:
        for i, l in enumerate(st.session_state.loads_list):
            if l['type'] == 'POINT': txt = f"{i+1}. POINT: {l['mag']}kN @ {l['loc']}m"
            elif l['type'] == 'UDL': txt = f"{i+1}. UDL: {l['mag']}kN/m ({l['start']}-{l['end']}m)"
            else: txt = f"{i+1}. UVL: {l['start_mag']}->{l['end_mag']} ({l['start']}-{l['end']}m)"
            
            c1, c2 = st.columns([4, 1])
            c1.text(txt)
            if c2.button("Del", key=f"del_{i}"):
                st.session_state.loads_list.pop(i)
                st.rerun()
                
        if st.button("Clear All Loads"):
            st.session_state.loads_list = []
            st.rerun()
    else:
        st.info("No loads added yet.")

# === MAIN AREA (VISUALIZATION) ===
with col_viz:
    # Run Analysis
    SupB_Calc = SupB if b_type != "Cantilever" else L
    x, V, M, Ra, Rb, Ma = calculate_analysis(L, SupA, SupB_Calc, b_type, st.session_state.loads_list)
    
    # 1. VISUAL PREVIEW (Using Matplotlib)
    st.subheader("Beam Preview")
    plt.style.use('dark_background')
    fig0 = Figure(figsize=(8, 2), dpi=100, facecolor='#0e1117')
    ax0 = fig0.add_subplot(111)
    ax0.set_facecolor('#0e1117'); ax0.axis('off')
    
    # Draw Beam & Supports
    ax0.plot([0, L], [0, 0], color='#9e9e9e', lw=6, solid_capstyle='round')
    ax0.set_xlim(-L*0.1, L*1.1); ax0.set_ylim(-2, 4)
    if b_type == "Cantilever":
        ax0.plot([SupA, SupA], [-1, 1], color='#b0bec5', lw=6)
    else:
        ax0.plot(SupA, -0.2, marker='^', markersize=10, color='#b0bec5', mec='white')
        ax0.plot(SupB, -0.2, marker='o', markersize=8, color='#b0bec5', mec='white')
        
    # Draw Loads
    for l in st.session_state.loads_list:
        if l["type"] == "POINT":
            ax0.arrow(l["loc"], 1.5, 0, -1.2, head_width=L*0.02, fc='#ff5252', ec='#ff5252')
        elif l["type"] == "UDL":
            ax0.fill_between([l["start"], l["end"]], 0.3, 0.8, color='#03a9f4', alpha=0.5)
        elif l["type"] == "UVL":
            h1 = 0.3 + (l["start_mag"]/50); h2 = 0.3 + (l["end_mag"]/50)
            ax0.fill([l["start"], l["end"], l["end"], l["start"]], [0.3, 0.3, h2, h1], color='#ff9800', alpha=0.5)
            
    st.pyplot(fig0)
    
    # 2. ANALYSIS RESULTS
    st.success(f"**Reaction A:** {Ra:.2f} kN | **Reaction B:** {Rb:.2f} kN | **Max Moment:** {max(M, key=abs):.2f} kNm")
    
    # 3. SFD / BMD PLOTS
    tab_sfd, tab_bmd = st.tabs(["Shear Force (SFD)", "Bending Moment (BMD)"])
    
    with tab_sfd:
        fig1 = Figure(figsize=(8, 3), dpi=100, facecolor='#0e1117')
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor('#0e1117')
        ax1.plot(x, V, color='#4fc3f7', lw=2)
        ax1.fill_between(x, V, 0, color='#03a9f4', alpha=0.2)
        ax1.set_ylabel("Shear (kN)", color='white'); ax1.grid(True, alpha=0.2)
        ax1.tick_params(colors='white')
        st.pyplot(fig1)
        
    with tab_bmd:
        fig2 = Figure(figsize=(8, 3), dpi=100, facecolor='#0e1117')
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor('#0e1117')
        ax2.plot(x, M, color='#ffb74d', lw=2)
        ax2.fill_between(x, M, 0, color='#ff9800', alpha=0.2)
        ax2.set_ylabel("Moment (kNm)", color='white'); ax2.grid(True, alpha=0.2)
        ax2.tick_params(colors='white')
        st.pyplot(fig2)

    # 4. EXPORT
    df = pd.DataFrame({"X (m)": x, "Shear (kN)": V, "Moment (kNm)": M})
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Analysis CSV", data=csv, file_name="beam_analysis.csv", mime="text/csv")
