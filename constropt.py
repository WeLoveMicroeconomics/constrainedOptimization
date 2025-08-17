import streamlit as st
import numpy as np
from scipy.optimize import minimize
import sympy as sp

st.title("Constrained Optimization Solver")

def pretty_round(x, tol=1e-6):
    if abs(x - round(x)) < tol:
        return int(round(x))
    else:
        return round(float(x), 6)

# User inputs
n_vars = st.number_input("Number of variables", min_value=1, max_value=6, value=3, step=1)
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"])
objective_str = st.text_input("Objective function in variables x1, x2, ...", "sqrt(x1*x2)*x3")
n_cons = st.number_input("Number of inequality constraints (>= 0)", min_value=0, max_value=6, value=2, step=1)
constraints_str = [st.text_input(f"Constraint {i+1} (e.g. x1 + x2 - 1 >= 0)", "") for i in range(n_cons)]

if st.button("Solve"):

    x_syms = sp.symbols(f'x1:{n_vars+1}')
    
    try:
        obj_expr = sp.sympify(objective_str)
        obj_func = sp.lambdify(x_syms, obj_expr, modules=['numpy'])
    except Exception as e:
        st.error(f"Error parsing objective function: {e}")
        st.stop()

    cons = []
    cons_exprs = []  # keep functions for feasibility check
    for c_str in constraints_str:
        if c_str.strip() == "":
            continue
        if ">=" not in c_str:
            st.error("Constraints must be in the form 'expression >= 0'")
            st.stop()
        lhs, rhs = c_str.split(">=")
        try:
            cons_expr = sp.sympify(lhs) - sp.sympify(rhs)
            base_func = sp.lambdify(x_syms, cons_expr, modules=['numpy'])
            
            def cons_func(x, f=base_func):
                val = f(*x)
                return np.array(val).astype(float)
            
            cons.append({'type': 'ineq', 'fun': cons_func})
            cons_exprs.append(base_func)
        except Exception as e:
            st.error(f"Error parsing constraint '{c_str}': {e}")
            st.stop()

    def fun_to_minimize(x):
        val = obj_func(*x)
        if max_or_min == "Maximize":
            return -val
        else:
            return val

    # ---- FIXES START ----
    # Initial guess (mid-range values)
    x0 = np.array([10.0] * n_vars)

    # Nonnegativity bounds only
    bounds = [(0, None) for _ in range(n_vars)]

    # Feasibility safeguard
    def make_feasible(x0, cons_exprs, bounds):
        x = np.copy(x0)
        # enforce bounds
        for i, (lb, ub) in enumerate(bounds):
            if lb is not None and x[i] < lb:
                x[i] = lb + 1e-6
            if ub is not None and x[i] > ub:
                x[i] = ub - 1e-6
        # check constraints
        for f in cons_exprs:
            val = f(*x)
            if val < 0:  # violation
                # shift slightly towards feasibility
                x = x + 0.1 * np.random.rand(len(x))
        return x

    # diagnostic: check feasibility before adjustment
    feasible_before = True
    for f in cons_exprs:
        if f(*x0) < -1e-8:  # tolerance
            feasible_before = False
            break

    x0_feasible = make_feasible(x0, cons_exprs, bounds)

    feasible_after = True
    for f in cons_exprs:
        if f(*x0_feasible) < -1e-8:
            feasible_after = False
            break

    if feasible_before:
        st.info(f"Initial guess {x0} is feasible ✅")
    else:
        st.warning(f"Initial guess {x0} is NOT feasible ❌ — adjusted to {x0_feasible}")

    # use the feasible-adjusted guess
    x0 = x0_feasible

    options = {
        'ftol': 1e-9,
        'maxiter': 500
    }
    # ---- FIXES END ----

    res = minimize(
        fun_to_minimize,
        x0,
        constraints=cons,
        bounds=bounds,
        method='SLSQP',
        options=options
    )

    if res.success:
        sol = res.x
        val = res.fun if max_or_min == "Minimize" else -res.fun
        st.success(
            f"Optimal solution found:\n" +
            "\n".join([f"x{i+1} = {pretty_round(sol[i])}" for i in range(n_vars)]) +
            f"\nObjective value: {round(float(val),6)}"
        )
    else:
        st.error(f"Optimization failed: {res.message}")
