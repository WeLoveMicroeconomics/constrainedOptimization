import streamlit as st
import numpy as np
from sympy import symbols, sympify, diff, lambdify
from scipy.optimize import minimize

st.title("Optimization Solver with Kuhn-Tucker Conditions")

# User selects number of decision variables
n_vars = st.slider("Number of decision variables", min_value=1, max_value=6, value=2)

# Create variables x1 to xn
x_vars = symbols(f'x1:{n_vars+1}')  # x1, x2, ..., xn

# Input objective function
st.subheader("Objective Function")
obj_type = st.selectbox("Optimization Type", ["Minimize", "Maximize"])
objective_str = st.text_input("Enter the objective function (use variables x1 to x6)", "x1**2 + x2**2")

# Input constraints
st.subheader("Constraints")
num_constraints = st.slider("Number of constraints (up to 6)", 0, 6, 2)
constraint_list = []
for i in range(num_constraints):
    constraint_str = st.text_input(f"Constraint {i+1}", f"x1 + x2 - {i+1} <= 0")
    constraint_list.append(constraint_str)

show_kkt = st.checkbox("Show Kuhn-Tucker Conditions")

if st.button("Solve"):
    # Parse the objective function
    try:
        f_expr = sympify(objective_str)
    except Exception as e:
        st.error(f"Invalid objective function: {e}")
        st.stop()

    if obj_type == "Maximize":
        f_expr = -f_expr

    # Convert to lambda function
    f_lambdified = lambdify(x_vars, f_expr, modules="numpy")

    # Initial guess
    x0 = np.ones(n_vars)

    # Parse constraints
    scipy_constraints = []
    parsed_constraints = []

    for con_str in constraint_list:
        try:
            if "<=" in con_str:
                lhs, rhs = con_str.split("<=")
                expr = sympify(lhs) - sympify(rhs)
                parsed_constraints.append(("ineq", expr))
                func = lambda x, e=expr: float(e.subs(dict(zip(x_vars, x))))
                scipy_constraints.append({'type': 'ineq', 'fun': func})
            elif ">=" in con_str:
                lhs, rhs = con_str.split(">=")
                expr = sympify(lhs) - sympify(rhs)
                parsed_constraints.append(("ineq", -expr))
                func = lambda x, e=-expr: float(e.subs(dict(zip(x_vars, x))))
                scipy_constraints.append({'type': 'ineq', 'fun': func})
            else:
                st.error("Each constraint must use '<=' or '>='.")
                st.stop()
        except Exception as e:
            st.error(f"Error in constraint: {e}")
            st.stop()

    # Run the optimizer
    try:
        result = minimize(lambda x: f_lambdified(*x), x0, constraints=scipy_constraints)
    except Exception as e:
        st.error(f"Optimization error: {e}")
        st.stop()

    if result.success:
        st.success("Optimization successful!")
        val = -result.fun if obj_type == "Maximize" else result.fun
        st.write(f"**Optimal value:** {val:.4f}")
        st.write("**Optimal solution:**")
        for i in range(n_vars):
            st.write(f"x{i+1} = {result.x[i]:.4f}")
    else:
        st.error("Optimization failed.")
        st.write(result.message)

    # KKT Conditions
    if show_kkt:
        st.subheader("Kuhn-Tucker Conditions")
        grad_f = [diff(f_expr, var) for var in x_vars]
        st.markdown("**1. Stationarity**")
        for i, g in enumerate(grad_f):
            st.latex(f"\\frac{{\\partial f}}{{\\partial x_{i+1}}} = {g}")

        st.markdown("**2. Primal Feasibility**")
        for idx, (_, g) in enumerate(parsed_constraints):
            st.latex(f"g_{idx+1}(x) = {g} \\leq 0")

        st.markdown("**3. Dual Feasibility**")
        st.latex("\\lambda_i \\geq 0")

        st.markdown("**4. Complementary Slackness**")
        st.latex("\\lambda_i \\cdot g_i(x) = 0")

        st.info("Lagrange multipliers not computed numerically in this version.")
