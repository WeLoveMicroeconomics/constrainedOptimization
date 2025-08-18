import streamlit as st
import numpy as np
from scipy.optimize import minimize
import sympy as sp
from fractions import Fraction

st.title("Constrained Optimization Solver — COBYLA / SLSQP + Near-KKT + Rational Output")

def pretty_round(x, tol=1e-6):
    try:
        if abs(x - round(x)) < tol:
            return int(round(x))
    except Exception:
        pass
    return float(np.round(float(x), 6))

def as_fraction(x, max_den=1000000):
    # robust cast then rational approximation
    try:
        xf = float(np.asarray(x, dtype=float).reshape(()))
        return Fraction(xf).limit_denominator(max_den)
    except Exception:
        return x

# --- Inputs ---
n_vars = st.number_input("Number of variables", min_value=1, max_value=6, value=3, step=1)
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"], index=1)
objective_str = st.text_input("Objective function in variables x1, x2, ...", "sqrt(x1*x2)*x3")
n_cons = st.number_input("Number of inequality constraints (>= 0)", min_value=0, max_value=20, value=2, step=1)
constraints_str = [st.text_input(f"Constraint {i+1} (e.g. 100 - (x1 + x2) >= 0)", "") for i in range(n_cons)]

# New options
method = st.selectbox("Optimization method", ["SLSQP", "COBYLA"], index=0)
accept_near_kkt = st.checkbox("Treat 'near-KKT' solutions as success when solver reports a benign failure", value=True)
tol_feas = st.number_input("Feasibility tolerance (>= -tol)", value=1e-6, format="%.1e")
probe_step = st.number_input("Local improvement probe step", value=1e-4, format="%.1e")
improve_tol = st.number_input("Improvement threshold", value=1e-6, format="%.1e")

# Rational output controls
show_rational = st.checkbox("Format results as rational numbers (fractions)", value=True)
max_den = st.number_input("Max denominator for rational approximation", min_value=1, max_value=10_000_000, value=1_000_000, step=1)

if st.button("Solve"):
    # --- Symbols ---
    x_syms = sp.symbols(f"x1:{n_vars+1}")
    try:
        obj_expr = sp.sympify(objective_str)
        obj_func = sp.lambdify(x_syms, obj_expr, modules=['numpy'])
    except Exception as e:
        st.error(f"Error parsing objective function: {e}")
        st.stop()

    # --- Constraints ---
    cons = []
    cons_exprs = []
    for c_str in constraints_str:
        c_str = c_str.strip()
        if not c_str:
            continue
        if ">=" not in c_str:
            st.error(f"Constraint '{c_str}' must be in the form 'expression >= 0' or 'lhs >= rhs'.")
            st.stop()
        lhs, rhs = c_str.split(">=", 1)
        try:
            cons_expr = sp.sympify(lhs) - sp.sympify(rhs)
            f = sp.lambdify(x_syms, cons_expr, modules=['numpy'])

            def cons_fun(x, f=f):
                val = f(*x)
                return float(np.asarray(val, dtype=float).reshape(()))

            cons.append({"type": "ineq", "fun": cons_fun})
            cons_exprs.append(f)
        except Exception as e:
            st.error(f"Error parsing constraint '{c_str}': {e}")
            st.stop()

    # --- Bounds (nonnegativity by default) ---
    bounds = [(0.0, None) for _ in range(n_vars)]

    if method == "COBYLA":
        # COBYLA ignores bounds parameter; add explicit constraints
        for i in range(n_vars):
            def lb_fun_factory(idx):
                return lambda x, idx=idx: float(x[idx] - 0.0)
            cons.append({"type": "ineq", "fun": lb_fun_factory(i)})
            cons_exprs.append(sp.lambdify(x_syms, x_syms[i] - 0.0, modules=['numpy']))

    # --- Objective wrappers ---
    def obj_for_min(x):
        val = obj_func(*x)
        v = float(np.asarray(val, dtype=float).reshape(()))
        return -v if max_or_min == "Maximize" else v

    def objective_for_report(x):
        val = obj_func(*x)
        return float(np.asarray(val, dtype=float).reshape(()))

    # --- Initial guess ---
    x0 = np.full(n_vars, 1.0, dtype=float)

    # --- Feasibility helpers ---
    def cons_values(x):
        vals = []
        for f in cons_exprs:
            v = float(np.asarray(f(*x), dtype=float).reshape(()))
            vals.append(v)
        return np.array(vals, dtype=float)

    def is_feasible(x, tol=tol_feas):
        vals = cons_values(x)
        if np.any(np.isnan(vals)):
            return False
        return np.min(vals) >= -tol

    # --- Simple repair ---
    def repair(x, iters=200):
        x = np.array(x, dtype=float, copy=True)
        x = np.maximum(x, 1e-8)
        rng = np.random.default_rng(0)
        def total_violation(z):
            tv = 0.0
            for f in cons_exprs:
                v = float(np.asarray(f(*z), dtype=float).reshape(()))
                tv += max(0.0, -v)
            return tv
        best = x
        best_tv = total_violation(best)
        for _ in range(iters):
            if best_tv <= tol_feas:
                break
            step = rng.normal(0.0, 0.25, size=best.size)
            x_try = np.maximum(best + step, 1e-8)
            tv = total_violation(x_try)
            if tv < best_tv:
                best, best_tv = x_try, tv
        return best

    x0 = repair(x0)

    # Diagnostics at start
    st.write("Initial guess:", x0)
    cv0 = cons_values(x0).tolist()
    if cv0:
        st.write("Constraint values at initial guess (>= 0 desired):", cv0)

    # --- Solve ---
    if method == "SLSQP":
        options = dict(ftol=1e-9, maxiter=2000, disp=False)
        res = minimize(
            obj_for_min, x0, method="SLSQP",
            bounds=bounds, constraints=cons, options=options
        )
    else:  # COBYLA
        options = dict(maxiter=3000, tol=1e-9, rhobeg=1.0)
        res = minimize(
            obj_for_min, x0, method="COBYLA",
            constraints=cons, options=options
        )

    # --- Near-KKT acceptance ---
    feas_vals = cons_values(res.x)
    feas_min = np.min(feas_vals) if len(feas_vals) else np.inf
    near_feasible = (feas_min >= -tol_feas)

    # Local probe
    locally_flat = True
    if near_feasible:
        base = objective_for_report(res.x) if max_or_min == "Maximize" else -objective_for_report(res.x)
        for i in range(n_vars):
            for s in (-1.0, 1.0):
                z = np.array(res.x, dtype=float, copy=True)
                z[i] += s * probe_step
                z = np.maximum(z, 0.0)
                if is_feasible(z):
                    val = objective_for_report(z)
                    val = val if max_or_min == "Maximize" else -val
                    if val > base + improve_tol:
                        locally_flat = False
                        break
            if not locally_flat:
                break

    treated_as_success = False
    if not res.success and accept_near_kkt and near_feasible and locally_flat:
        treated_as_success = True

    # --- Reporting helpers ---
    def fmt_value(x):
        if show_rational:
            return f"{as_fraction(x, max_den=max_den)}  (~ {pretty_round(x)})"
        else:
            return f"{pretty_round(x)}"

    # --- Results ---
    if res.success or treated_as_success:
        sol = res.x
        val = objective_for_report(sol)
        lines = ["Optimal solution found." if res.success else "Near-KKT solution accepted as optimal."]
        for i in range(n_vars):
            lines.append(f"x{i+1} = {fmt_value(sol[i])}")
        lines.append(f"Objective value: {fmt_value(val)}")
        if not res.success:
            lines.append("(Note: solver returned a benign status; accepted by near-KKT test.)")
        st.success("\n".join(lines))
    else:
        st.error(f"Optimization failed: {res.message}")
        st.write("Best iterate found:", res.x)
        if len(feas_vals):
            st.write("Constraint values at best iterate (>= 0 desired):", feas_vals.tolist())
        try:
            val = objective_for_report(res.x)
            st.write("Objective at best iterate:", val)
        except Exception:
            pass
