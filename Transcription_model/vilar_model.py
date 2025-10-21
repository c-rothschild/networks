
from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional

@dataclass(frozen=True)
class VilarParams:
    alpha_A: float
    alphaA_act: float
    alpha_R: float
    alphaR_act: float
    beta_A: float
    beta_R: float
    delta_MA: float
    delta_MR: float
    delta_A: float
    delta_R: float
    kon_DA: float
    kon_DR: float
    kon_C: float
    koff_DA: float
    koff_DR: float

def default_params() -> VilarParams:
    return VilarParams(
        alpha_A=50.0,
        alphaA_act=500.0,
        alpha_R=0.01,
        alphaR_act=50.0,
        beta_A=50.0,
        beta_R=5.0,
        delta_MA=10.0,
        delta_MR=0.5,
        delta_A=1.0,
        delta_R=0.2,
        kon_DA=1.0,
        kon_DR=1.0,
        kon_C=2.0,
        koff_DA=50.0,
        koff_DR=100.0,
    )

def vilar_full(t: float, s: List[float], p: Optional[VilarParams] = None) -> List[float]:
    if p is None:
        p = default_params()
    DA, DAp, DR, DRp, MA, MR, A, R, C = s
    bind_DA = p.kon_DA * DA * A
    unbind_DA = p.koff_DA * DAp
    bind_DR = p.kon_DR * DR * A
    unbind_DR = p.koff_DR * DRp
    tx_A = p.alpha_A * DA + p.alphaA_act * DAp
    tx_R = p.alpha_R * DR + p.alphaR_act * DRp
    tl_A = p.beta_A * MA
    tl_R = p.beta_R * MR
    deg_MA = p.delta_MA * MA
    deg_MR = p.delta_MR * MR
    deg_A_free = p.delta_A * A
    deg_R_free = p.delta_R * R
    form_C = p.kon_C * A * R
    loss_C_A = p.delta_A * C  # A degrades in complex -> releases R
    loss_C_R = p.delta_R * C  # R degrades in complex -> consumes complex
    dDA_dt  = -bind_DA + unbind_DA
    dDAp_dt =  bind_DA - unbind_DA
    dDR_dt  = -bind_DR + unbind_DR
    dDRp_dt =  bind_DR - unbind_DR
    dMA_dt = tx_A - deg_MA
    dMR_dt = tx_R - deg_MR
    dA_dt = tl_A - bind_DA + unbind_DA - bind_DR + unbind_DR - form_C - deg_A_free
    dR_dt = tl_R - form_C + loss_C_A - deg_R_free
    dC_dt = form_C - loss_C_A - loss_C_R
    return [dDA_dt, dDAp_dt, dDR_dt, dDRp_dt, dMA_dt, dMR_dt, dA_dt, dR_dt, dC_dt]

# Optional: SciPy-based simulator for stiff systems
def simulate_scipy(rhs, y0, t0, t1, max_step=0.1, rtol=1e-6, atol=1e-9):
    try:
        from scipy.integrate import solve_ivp
    except Exception as e:
        raise RuntimeError("SciPy is required for simulate_scipy but not available") from e
    sol = solve_ivp(rhs, (t0, t1), y0, method="Radau", max_step=max_step, rtol=rtol, atol=atol)
    return sol.t, sol.y
