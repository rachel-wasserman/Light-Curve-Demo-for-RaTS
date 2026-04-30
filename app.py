import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Light-Curve Detection Demo", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    div[data-testid="stMarkdownContainer"] {
        font-size: 22px;
    }

    /* Move the red value label above the slider thumb */
    div[data-testid="stSlider"] [data-testid="stWidgetLabel"] + div div {
        overflow: visible !important;
    }

    div[data-testid="stSlider"] div[style*="transform"] {
        transform: translateY(-18px) !important;
        z-index: 10 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Defaults
# -----------------------
F0_default = 1.0
tau_default = 1.0
threshold = 0.4

alpha1 = 2.0
alpha2 = 1.5

obs_windows = [
    (0.5, 0.7),
    (1.8, 2.0),
    (3.2, 3.4),
    (5.0, 5.2),
]

t = np.linspace(0, 7, 3000)

# -----------------------
# Light-curve models
# -----------------------
def tophat(t_rel, F0, tau):
    return np.where((t_rel > 0) & (t_rel <= tau), F0, 0.0)

def fred(t_rel, F0, tau):
    return np.where(t_rel > 0, F0 * np.exp(-t_rel / tau), 0.0)

def sbpl(t_rel, F0, tau, alpha1, alpha2):
    flux = np.zeros_like(t_rel)

    rise = (t_rel > 0) & (t_rel <= tau)
    decay = t_rel > tau

    flux[rise] = F0 * (t_rel[rise] / tau) ** alpha1
    flux[decay] = F0 * (t_rel[decay] / tau) ** (-alpha2)

    return flux

def average_flux_in_window(t, flux, start, end):
    mask = (t >= start) & (t <= end)

    if np.sum(mask) < 2:
        return 0.0

    integral = np.trapezoid(flux[mask], t[mask])
    return integral / (end - start)

def detection_fraction_over_start_times(
    F0_current,
    tau_current,
    alpha1,
    alpha2,
    obs_windows,
    threshold,
    start_min=-1.0,
    start_max=6.0,
    start_step=0.05
):
    start_times = np.arange(start_min, start_max + start_step, start_step)

    detected_counts = {
        "Tophat": 0,
        "FRED": 0,
        "SBPL": 0
    }

    total = len(start_times)

    for start_time in start_times:
        t_rel = t - start_time

        flux_tophat = tophat(t_rel, F0_current, tau_current)
        flux_fred = fred(t_rel, F0_current, tau_current)
        flux_sbpl = sbpl(t_rel, F0_current, tau_current, alpha1, alpha2)

        for name, flux in [
            ("Tophat", flux_tophat),
            ("FRED", flux_fred),
            ("SBPL", flux_sbpl),
        ]:
            detected = False

            for start, end in obs_windows:
                avg_flux = average_flux_in_window(t, flux, start, end)

                if avg_flux > threshold:
                    detected = True
                    break

            if detected:
                detected_counts[name] += 1

    fractions = {
        name: detected_counts[name] / total
        for name in detected_counts
    }

    return fractions, detected_counts, total

# -----------------------
# App layout
# -----------------------
st.title("Interpreting Light-Curve Shape Effects")

col1, col2, col3 = st.columns(3)

with col1:
    start_time = st.slider(r"$t_0$ (days)", -1.0, 6.0, 0.0, 0.05)

with col2:
    F0_current = st.slider(r"$F_0$ (Jy)", 0.05, 2.0, F0_default, 0.05)

with col3:
    tau_current = st.slider(r"$\tau$ (days)", 0.1, 5.0, tau_default, 0.05)

# -----------------------
# Compute current curves
# -----------------------
t_rel = t - start_time

flux_tophat = tophat(t_rel, F0_current, tau_current)
flux_fred = fred(t_rel, F0_current, tau_current)
flux_sbpl = sbpl(t_rel, F0_current, tau_current, alpha1, alpha2)

fractions, counts, total = detection_fraction_over_start_times(
    F0_current=F0_current,
    tau_current=tau_current,
    alpha1=alpha1,
    alpha2=alpha2,
    obs_windows=obs_windows,
    threshold=threshold,
)

# -----------------------
# Main display layout
# -----------------------
plot_col, info_col = st.columns([2.2, 1])

with plot_col:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for i, (start, end) in enumerate(obs_windows):
        ax.axvspan(
            start,
            end,
            alpha=0.15,
            color="#7FA8D1",
            label="Observation Window" if i == 0 else None
        )

    ax.plot(t, flux_tophat, label="Tophat", linewidth=2, color="#7FA8D1")
    ax.plot(t, flux_fred, label="FRED", linewidth=2, color="#F4B6C2")
    ax.plot(t, flux_sbpl, label="SBPL", linewidth=2, color="#B39EB5")

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label="Detection Threshold",
        color="red"
    )

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Flux (Jy)", fontsize=11)
    ax.set_title("Light-Curve Shape and Detection", fontsize=13)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.2 * max(F0_current, threshold))
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(loc="upper right", fontsize=10)

    st.pyplot(fig, use_container_width=True)

with info_col:
    st.markdown("## Detection Summary")

    st.markdown("### Current Parameters")

    st.markdown(
        f"""
        <div style="font-size:30px; line-height:1.9;">
        <b>t<sub>0</sub>:</b> {start_time:.2f} days<br>
        <b>F<sub>0</sub>:</b> {F0_current:.2f} Jy<br>
        <b>τ:</b> {tau_current:.2f} days
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    st.markdown("### Detection Probability Over t<sub>0</sub>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="font-size:32px; line-height:1.8;">
        <b>Tophat:</b> {100*fractions['Tophat']:.1f}% <span style="font-size:26px;">({counts['Tophat']}/{total})</span><br>
        <b>FRED:</b> {100*fractions['FRED']:.1f}% <span style="font-size:26px;">({counts['FRED']}/{total})</span><br>
        <b>SBPL:</b> {100*fractions['SBPL']:.1f}% <span style="font-size:26px;">({counts['SBPL']}/{total})</span>
        </div>
        """,
        unsafe_allow_html=True
    )