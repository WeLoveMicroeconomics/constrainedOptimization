import streamlit as st
import numpy as np
from scipy.optimize import minimize
import sympy as sp

st.title("Constrained Optimization Solver (custom variables)")

def pretty_round(x, tol=1e-6):
    try:
        if abs(x - round(x)) < tol:
            return int(round(x))
    except Exception:
        pass
    return float(np.round(float(x), 6))

# --- Inputs ---
# Enter custom variable names (comma-separated). Examples: "x, y", "a, b, c", "x1, x2, x3"
var_names_str = st.text_input("Variable names (comma-separated)", "x, y, z")
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"])
objective_str = st.text_input("Objective function (use your variables)", "sqrt(x*y)*z")
n_cons = st.number_input("Number of inequality constraints (>= 0)", min_value=0, max_value=12, value=2, step=1)
constraints_str = [st.text_input(f"Constraint {i+1} (e.g. x + y - 1 >= 0 or lhs >= rhs)", "") for i in range(n_cons)]

if st.button("Solve"):
    # --- Parse & validate variable names ---
    var_names = [v.strip() for v in var_names_str.split(",") if v.strip()]
    if not var_names:
        st.error("Please provide at least one variable name (e.g., 'x' or 'x, y').")
        st.stop()
    # Check duplicates
    if len(set(var_names)) != len(var_names):
        st.error("Variable names must be unique. Please remove duplicates.")
        st.stop()
    # Create symbols (will error if a name is invalid)
    try:
        x_syms = sp.symbols(var_names)
    except Exception as e:
        st.error(f"Could not create symbols from your variable names: {e}")
        st.stop()

    n_vars = len(var_names)
    allowed_syms = set(x_syms)

    # Helper for friendly name display
    def var_label(i):
        return str(var_names[i])

    # robust lambdify namespace (supports Abs/Max/Min with NumPy)
    LAMBDA_MODULES = [
        {"Abs": np.abs, "Max": np.maximum, "Min": np.minimum},
        "numpy",
    ]

    def parse_and_validate(expr_str, what="expression"):
        """Parse string to sympy expr and ensure only allowed symbols are used."""
        try:
            expr = sp.sympify(expr_str)
        except Exception as e:
            st.error(f"Error parsing {what}: {e}")
            st.stop()
        extra = expr.free_symbols - allowed_syms
        if extra:
            names = ", ".join(sorted(str(s) for s in extra))
            st.error(
                f"{what.capitalize()} uses variables not listed in 'Variable names' "
                f"(extra: {names}). Add them to the list or remove them."
            )
            st.stop()
        return expr

    # --- Objective ---
    obj_expr = parse_and_validate(objective_str, what="objective")
    try:
        obj_func = sp.lambdify(x_syms, obj_expr, modules=LAMBDA_MODULES)
    except Exception as e:
        st.error(f"Error compiling objective: {e}")
        st.stop()

    # --- Constraints ---
    cons = []
    cons_exprs = []
    for idx, c_str in enumerate(constraints_str, start=1):
        c_str = c_str.strip()
        if not c_str:
            continue
        if ">=" not in c_str:
            st.error(f"Constraint {idx} ('{c_str}') must be 'expression >= 0' or 'lhs >= rhs'.")
            st.stop()
        lhs_str, rhs_str = c_str.split(">=", 1)
        lhs = parse_and_validate(lhs_str, what=f"constraint {idx} (lhs)")
        rhs = parse_and_validate(rhs_str, what=f"constraint {idx} (rhs)")
        cons_expr = lhs - rhs

        try:
            f_num = sp.lambdify(x_syms, cons_expr, modules=LAMBDA_MODULES)
        except Exception as e:
            st.error(f"Error compiling constraint {idx}: {e}")
            st.stop()

        def cons_fun(x, f=f_num):
            val = f(*x)
            return float(np.asarray(val, dtype=float).reshape(()))  # SLSQP wants float

        cons.append({"type": "ineq", "fun": cons_fun})
        cons_exprs.append(f_num)

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
            try:
                val = float(np.asarray(f(*x), dtype=float).reshape(()))
            except Exception:
                return False
            if np.isnan(val) or val < -eps:
                return False
        return True

    def total_violation(z):
        tv = 0.0
        for f in cons_exprs:
            try:
                v = float(np.asarray(f(*z), dtype=float).reshape(()))
            except Exception:
                v = -np.inf  # force violation to be "bad"
            tv += max(0.0, -v)
        return tv

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
            lows = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
            highs = np.array([b[1] if b[1] is not None else  np.inf for b in bounds], dtype=float)
            x_try = np.clip(x + step, lows, highs)
            if total_violation(x_try) < total_violation(x):
                x = x_try
        return x  # may still be infeasible; SLSQP can often recover

    x0 = repair(x0)

    # Diagnostics
    cons_vals0 = []
    for f in cons_exprs:
        cons_vals0.append(float(np.asarray(f(*x0), dtype=float).reshape(())))
    st.write("Variables:", var_names)
    st.write("Initial guess (in your variable order):", {var_names[i]: x0[i] for i in range(n_vars)})
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
            + "\n".join([f"{var_label(i)} = {pretty_round(sol[i])}" for i in range(n_vars)])
            + f"\nObjective value: {pretty_round(val)}"
        )
    else:
        st.error(f"Optimization failed: {res.message}")
        st.write("Best iterate found:", {var_names[i]: float(res.x[i]) for i in range(n_vars)})
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
