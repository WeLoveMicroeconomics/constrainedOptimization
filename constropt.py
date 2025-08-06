import streamlit as st
import numpy as np
from sympy import symbols, sympify, diff, lambdify

st.title("Constrained Optimization Solver with Kuhn-Tucker Conditions")

# Select number of variables
n_vars = st.slider("Number of decision variables", min_value=1, max_value=6, value=2)
x_vars = symbols(f'x1:{n_vars+1}')  # creates x1, x2, ..., xn

# Objective function input
st.subheader("Objective Function")
obj_type = st.selectbox("Optimization Type", ["Minimize", "Maximize"])
objective_str = st.text_input("Enter objective function (e.g. x1**2 + x2**2)", "x1**2 + x2**2")

# Constraints input
st.subheader("Constraints")
num_constraints = st.slider("Number of constraints (up to 6)", 0, 6, 2)
constraint_list = []
for i in range(num_constraints):
    constraint_str = st.text_input(f"Constraint {i+1} (e.g. x1 + x2 - 1 <= 0)", f"x1 + x2 - {i+1} <= 0")
    constraint_list.append(constraint_str)

show_kkt = st.checkbox("Show Kuhn-Tucker Conditions")

if st.button("Solve"):

    # Parse the objective function
    try:
        f_expr = sympify(objective_str)
        if obj_type == "Maximize":
            f_expr = -f_expr
        f_func = lambdify([x_vars], f_expr, modules='numpy')
    except Exception as e:
        st.error(f"Invalid objective function: {e}")
        st.stop()

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
                func = lambdify([x_vars], expr, modules='numpy')
                scipy_constraints.append({'type': 'ineq', 'fun': func})
                parsed_constraints.append(expr)
            elif ">=" in con_str:
                lhs, rhs = con_str.split(">=")
                expr = sympify(rhs) - sympify(lhs)  # flip to ≤ 0
                func = lambdify([x_vars], expr, modules='numpy')
                scipy_constraints.append({'type': 'ineq', 'fun': func})
                parsed_constraints.append(expr)
            else:
                st.error("Constraint must contain '<=' or '>='")
                st.stop()
        except Exception as e:
            st.error(f"Error parsing constraint: {e}")
            st.stop()

    # Run the optimizer
    try:
        result = minimize(f_func, x0, constraints=scipy_constraints)
    except Exception as e:
        st.error(f"Optimization error: {e}")
        st.stop()

    # Display results
    if result.success:
        st.success("Optimization successful!")
        optimal_value = -result.fun if obj_type == "Maximize" else result.fun
        st.write(f"**Optimal value:** {optimal_value:.4f}")
        st.write("**Solution:**")
        for i in range(n_vars):
            st.write(f"x{i+1} = {result.x[i]:.4f}")
    else:
        st.error("Optimization failed.")
        st.write(result.message)

    # Show Kuhn-Tucker Conditions
    if show_kkt:
        st.subheader("Kuhn-Tucker Conditions")

        st.markdown("**1. Stationarity**")
        grad_f = [diff(f_expr, var) for var in x_vars]
        for i, g in enumerate(grad_f):
            st.latex(f"\\frac{{\\partial f}}{{\\partial x_{i+1}}} = {g}")

        st.markdown("**2. Primal Feasibility**")
        for idx, g in enumerate(parsed_constraints):
            st.latex(f"g_{{{idx+1}}}(x) = {g} \\leq 0")

        st.markdown("**3. Dual Feasibility**")
        st.latex("\\lambda_i \\geq 0")

        st.markdown("**4. Complementary Slackness**")
        st.latex("\\lambda_i \\cdot g_i(x) = 0")

        st.info("Note: This version does not compute Lagrange multipliers numerically.")
