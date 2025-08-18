import streamlit as st
import numpy as np
from scipy.optimize import minimize
import sympy as sp

st.title("Constrained Optimization Solver (fixed)")

def pretty_round(x, tol=1e-6):
    try:
        if abs(x - round(x)) < tol:
            return int(round(x))
    except Exception:
        pass
    return float(np.round(float(x), 6))

# --- Inputs ---
n_vars = st.number_input("Number of variables", min_value=1, max_value=6, value=3, step=1)
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"])
objective_str = st.text_input("Objective function in variables x1, x2, ...", "sqrt(x1*x2)*x3")
n_cons = st.number_input("Number of inequality constraints (>= 0)", min_value=0, max_value=12, value=2, step=1)
constraints_str = [st.text_input(f"Constraint {i+1} (e.g. x1 + x2 - 1 >= 0)", "") for i in range(n_cons)]

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
                # Ensure scalar float return (SLSQP expects float or 1-D array)
                return float(np.asarray(val, dtype=float).reshape(()))

            cons.append({"type": "ineq", "fun": cons_fun})
            cons_exprs.append(f)
        except Exception as e:
            st.error(f"Error parsing constraint '{c_str}': {e}")
            st.stop()

    # --- Objective wrapper (always return float) ---
    def obj_for_min(x):
        val = obj_func(*x)
        v = float(np.asarray(val, dtype=float).reshape(()))
        return -v if max_or_min == "Maximize" else v

    # --- Bounds (nonnegativity by default) ---
    bounds = [(0.0, None) for _ in range(n_vars)]

    # --- Initial guess ---
    x0 = np.full(n_vars, 1.0, dtype=float)

    # --- Feasibility repair heuristic ---
    def is_feasible(x, eps=1e-9):
        for f in cons_exprs:
            val = float(np.asarray(f(*x), dtype=float).reshape(()))
            if np.isnan(val) or val < -eps:
                return False
        return True

    def repair(x, iters=200):
        x = np.array(x, dtype=float, copy=True)
        # project to bounds
        for i, (lb, ub) in enumerate(bounds):
            if lb is not None and (np.isnan(x[i]) or x[i] < lb):
                x[i] = lb + 1e-6
            if ub is not None and x[i] > ub:
                x[i] = ub - 1e-6
        # try small random nudges to satisfy constraints
        rng = np.random.default_rng(0)
        for _ in range(iters):
            if is_feasible(x):
                return x
            step = rng.normal(0.0, 0.25, size=x.size)
            x_try = np.clip(x + step, [b[0] if b[0] is not None else -np.inf for b in bounds],
                                  [b[1] if b[1] is not None else  np.inf for b in bounds])
            # accept if improves total violation
            def total_violation(z):
                tv = 0.0
                for f in cons_exprs:
                    v = float(np.asarray(f(*z), dtype=float).reshape(()))
                    tv += max(0.0, -v)
                return tv
            if total_violation(x_try) < total_violation(x):
                x = x_try
        return x  # may still be infeasible; SLSQP can often recover

    x0 = repair(x0)

    # Diagnostics
    cons_vals0 = []
    for f in cons_exprs:
        cons_vals0.append(float(np.asarray(f(*x0), dtype=float).reshape(())))
    st.write("Initial guess:", x0)
    if cons_vals0:
        st.write("Constraint values at initial guess (>=0 desired):", cons_vals0)

    # --- Solve ---
    options = dict(ftol=1e-9, maxiter=1000, disp=False)
    try:
        res = minimize(
            obj_for_min,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options=options,
        )
    except Exception as e:
        st.error(f"SciPy failed to start the optimizer: {e}")
        st.stop()

    # --- Results ---
    if res.success:
        sol = res.x
        val = obj_for_min(sol)
        if max_or_min == "Maximize":
            val = -val
        st.success(
            "Optimal solution found:\n"
            + "\n".join([f"x{i+1} = {pretty_round(sol[i])}" for i in range(n_vars)])
            + f"\nObjective value: {pretty_round(val)}"
        )
    else:
        st.error(f"Optimization failed: {res.message}")
        st.write("Best iterate found:", res.x)
        # Show constraint values at best iterate
        viols = []
        for f in cons_exprs:
            viols.append(float(np.asarray(f(*res.x), dtype=float).reshape(())))
        if viols:
            st.write("Constraint values at best iterate (>=0 desired):", viols)
        # Try to show objective at best iterate if finite
        try:
            val = obj_for_min(res.x)
            st.write("Objective (per your selection):", -val if max_or_min == "Maximize" else val)
        except Exception:
            pass
