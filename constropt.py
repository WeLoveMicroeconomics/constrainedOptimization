import streamlit as st
import numpy as np
from scipy.optimize import minimize
import sympy as sp

st.title("Constrained Optimization Solver")

n_vars = st.number_input("Number of variables", min_value=1, max_value=6, value=2, step=1)
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"])
objective_str = st.text_input("Objective function in variables x1, x2, ...", "x1**2 + x2**2")
n_cons = st.number_input("Number of inequality constraints (>= 0)", min_value=0, max_value=6, value=1, step=1)
constraints_str = [st.text_input(f"Constraint {i+1} (e.g. x1 + x2 - 1 >= 0)", "") for i in range(n_cons)]

if st.button("Solve"):

    x_syms = sp.symbols(f'x1:{n_vars+1}')
    
    try:
        obj_expr = sp.sympify(objective_str)
        obj_func = sp.lambdify(x_syms, obj_expr, modules='numpy')
    except Exception as e:
        st.error(f"Error parsing objective function: {e}")
        st.stop()

    cons = []
    for c_str in constraints_str:
        if c_str.strip() == "":
            continue
        if ">=" not in c_str:
            st.error("Constraints must be in the form 'expression >= 0'")
            st.stop()
        lhs, rhs = c_str.split(">=")
        try:
            cons_expr = sp.sympify(lhs) - sp.sympify(rhs)
            cons_func = sp.lambdify(x_syms, cons_expr, modules='numpy')
            cons.append({'type': 'ineq', 'fun': cons_func})
        except Exception as e:
            st.error(f"Error parsing constraint '{c_str}': {e}")
            st.stop()

    def fun_to_minimize(x):
        val = obj_func(*x)
        if max_or_min == "Maximize":
            return -val
        else:
            return val

    x0 = np.zeros(n_vars)

    res = minimize(fun_to_minimize, x0, constraints=cons, method='SLSQP')

    if res.success:
        sol = res.x
        val = res.fun if max_or_min == "Minimize" else -res.fun
        st.success(f"Optimal solution found:\n" + 
                   "\n".join([f"x{i+1} = {sol[i]:.4f}" for i in range(n_vars)]) + 
                   f"\nObjective value: {val:.4f}")
    else:
        st.error(f"Optimization failed: {res.message}")
