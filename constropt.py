
import numpy as np
import streamlit as st
import sympy as sp
from scipy.optimize import minimize
from fractions import Fraction

st.set_page_config(page_title="Constrained Optimizer (Simple)", page_icon="✅", layout="centered")
st.title("Constrained Optimizer — Simple & Robust")

# ------------------ Helpers ------------------
def to_scalar(v):
    return float(np.asarray(v, dtype=float).reshape(()))

def pretty_round(x, tol=1e-9):
    try:
        if abs(x - round(x)) < tol:
            return int(round(x))
    except Exception:
        pass
    return float(np.round(float(x), 6))

def as_fraction(x, max_den=1_000_000):
    try:
        xf = float(np.asarray(x, dtype=float).reshape(()))
        return Fraction(xf).limit_denominator(max_den)
    except Exception:
        return x

# ------------------ UI ------------------
col1, col2 = st.columns(2)
with col1:
    n_vars = st.number_input("Number of variables", 1, 12, 3, 1)
    sense = st.selectbox("Problem type", ["Maximize", "Minimize"], index=0)
with col2:
    method = st.selectbox("Method", ["COBYLA", "SLSQP"], index=0)
    use_near_kkt = st.checkbox("Accept near-KKT solutions (treat benign failures as success)", value=True)

objective_str = st.text_input("Objective f(x1, x2, ...)", "sqrt(x1*x2)*x3")

n_cons = st.number_input("Number of inequality constraints (g(x) ≥ 0)", 0, 20, 2, 1)
constraints_str = [st.text_input(f"Constraint {i+1}", "") for i in range(n_cons)]

st.markdown("**Defaults:** nonnegativity bounds xᵢ ≥ 0, initial guess x₀ = 1 for all variables.")

with st.expander("Output formatting"):
    show_frac = st.checkbox("Show results as fractions", value=True)
    max_den = st.number_input("Max denominator", 1, 10_000_000, 1_000_000, 1)

if st.button("Solve"):
    # ---- Build symbols and objective ----
    x_syms = sp.symbols(" ".join([f"x{i+1}" for i in range(n_vars)]), real=True)
    try:
        obj_expr = sp.sympify(objective_str)
        obj_func = sp.lambdify(x_syms, obj_expr, modules=['numpy'])
    except Exception as e:
        st.error(f"Error parsing objective: {e}")
        st.stop()

    # ---- Parse constraints g(x) >= 0 ----
    cons = []
    cons_exprs = []  # symbolic versions for feasibility checks
    for s in constraints_str:
        s = s.strip()
        if not s:
            continue
        if ">=" in s:
            lhs, rhs = s.split(">=", 1)
            g_expr = sp.simplify(sp.sympify(lhs) - sp.sympify(rhs))
        else:
            # treat as already g(x) >= 0
            g_expr = sp.simplify(sp.sympify(s))
        f = sp.lambdify(x_syms, g_expr, modules=['numpy'])

        def g_fun(x, f=f):
            return to_scalar(f(*x))

        cons.append({"type": "ineq", "fun": g_fun})
        cons_exprs.append(f)

    # ---- Nonnegativity bounds ----
    bounds = [(0.0, None) for _ in range(n_vars)]
    if method == "COBYLA":
        # add x_i >= 0 explicitly since COBYLA ignores 'bounds'
        for i in range(n_vars):
            def lb_fun_factory(idx):
                return lambda x, idx=idx: float(x[idx])  # >= 0
            cons.append({"type": "ineq", "fun": lb_fun_factory(i)})
            cons_exprs.append(sp.lambdify(x_syms, x_syms[i], modules=['numpy']))

    # ---- Objective wrapper ----
    def obj_for_min(x):
        v = to_scalar(obj_func(*x))
        return -v if sense == "Maximize" else v

    def obj_for_report(x):
        return to_scalar(obj_func(*x))

    # ---- Initial guess & light repair ----
    x0 = np.full(n_vars, 1.0, dtype=float)
    def cons_values(x):
        return np.array([to_scalar(f(*x)) for f in cons_exprs], dtype=float) if cons_exprs else np.array([])

    def is_feasible(x, tol=1e-8):
        if not len(cons_exprs):
            return True
        vals = cons_values(x)
        if np.any(np.isnan(vals)):
            return False
        return np.min(vals) >= -tol

    def repair(x, iters=200):
        x = np.maximum(np.asarray(x, float), 1e-8)
        rng = np.random.default_rng(0)
        def total_violation(z):
            tv = 0.0
            for f in cons_exprs:
                tv += max(0.0, -to_scalar(f(*z)))
            return tv
        best = x.copy()
        best_tv = total_violation(best)
        for _ in range(iters):
            if best_tv <= 1e-8:
                break
            z = np.maximum(best + rng.normal(0, 0.25, size=best.size), 1e-8)
            tv = total_violation(z)
            if tv < best_tv:
                best, best_tv = z, tv
        return best

    x0 = repair(x0)

    st.write("Initial guess:", x0)
    if len(cons_exprs):
        st.write("Constraint values at x₀ (≥ 0 desired):", cons_values(x0).tolist())

    # ---- Optimize ----
    try:
        if method == "SLSQP":
            res = minimize(obj_for_min, x0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options=dict(ftol=1e-9, maxiter=2000, disp=False))
        else:
            res = minimize(obj_for_min, x0, method="COBYLA",
                           constraints=cons, options=dict(maxiter=3000, tol=1e-9, rhobeg=1.0))
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        st.stop()

    # ---- Near-KKT acceptance (simple) ----
    feas_vals = cons_values(res.x)
    near_feasible = (len(feas_vals) == 0) or (np.min(feas_vals) >= -1e-6)

    locally_flat = True
    if near_feasible:
        base = obj_for_report(res.x) if sense == "Maximize" else -obj_for_report(res.x)
        for i in range(n_vars):
            for sgn in (-1.0, 1.0):
                z = np.maximum(res.x.copy(), 0.0)
                z[i] = max(0.0, z[i] + sgn * 1e-4)
                if is_feasible(z):
                    val = obj_for_report(z)
                    val = val if sense == "Maximize" else -val
                    if val > base + 1e-6:
                        locally_flat = False
                        break
            if not locally_flat:
                break

    accept = res.success or (use_near_kkt and near_feasible and locally_flat)

    # ---- Reporting ----
    def fmt(x):
        if show_frac:
            return f"{as_fraction(x, max_den=max_den)}  (~{pretty_round(x)})"
        return f"{pretty_round(x)}"

    if accept:
        sol = res.x
        val = obj_for_report(sol)
        lines = []
        lines.append("**Solution**" + (" (near-KKT accepted)" if (not res.success) else ""))
        for i in range(n_vars):
            lines.append(f"x{i+1} = {fmt(sol[i])}")
        lines.append(f"Objective value = {fmt(val)}")
        st.success("\n".join(lines))
    else:
        st.error(f"Optimization failed: {res.message}")
        st.write("Best iterate:", res.x)
        if len(feas_vals):
            st.write("Constraint values at best iterate (≥ 0 desired):", feas_vals.tolist())
        try:
            st.write("Objective at best iterate:", obj_for_report(res.x))
        except Exception:
            pass
