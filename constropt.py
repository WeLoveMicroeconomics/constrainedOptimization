import streamlit as st
import numpy as np
from sympy import symbols, sympify, diff, Eq, solve
from scipy.optimize import minimize

# Streamlit interface
st.title("Optimization Solver with Kuhn-Tucker Conditions")

st.markdown("### Step 1: Objective Function")
obj_type = st.selectbox("Optimization Type", ["Minimize", "Maximize"])
objective_str = st.text_input("Enter the objective function (in terms of x1 to x6)", "x1**2 + x2**2")

st.markdown("### Step 2: Constraints")
num_constraints = st.slider("Number of constraints (up to 6)", 0, 6, 2)

constraint_list = []
for i in range(num_constraints):
    constraint_str = st.text_input(f"Constraint {i+1} (format: expression <= 0 or expression >= 0)", f"x1 + x2 - {i+1} <= 0")
    constraint_list.append(constraint_str)

show_kkt = st.checkbox("Show Kuhn-Tucker Conditions")

if st.button("Solve"):
    # Define variables
    x = symbols("x1 x2 x3 x4 x5 x6")
    n_vars = 6
    x_vars = x[:n_vars]

    # Parse objective function
    try:
        f_expr = sympify(objective_str)
    except Exception as e:
        st.error(f"Invalid objective function: {e}")
        st.stop()

    # Determine if minimizing or maximizing
    if obj_type == "Maximize":
        f_expr = -f_expr  # convert to minimization

    # Parse constraints
    parsed_constraints = []
    scipy_constraints = []

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
                expr = sympify(sympify(lhs) - sympify(rhs))
                parsed_constraints.append(("ineq", -expr))
                func = lambda x, e=-expr: float(e.subs(dict(zip(x_vars, x))))
                scipy_constraints.append({'type': 'ineq', 'fun': func})
            else:
                st.error("Each constraint must include '<=' or '>='")
                st.stop()
        except Exception as e:
            st.error(f"Error in constraint {con_str}: {e}")
            st.stop()

    # Convert objective to function
    f_lambdified = lambda x_val: float(f_expr.subs(dict(zip(x_vars, x_val))))

    # Run optimizer
    x0 = np.ones(n_vars)  # initial guess
    result = minimize(f_lambdified, x0, constraints=scipy_constraints)

    if result.success:
        solution = result.x
        st.success("Optimization successful!")
        if obj_type == "Maximize":
            st.write(f"**Maximum Value:** {-result.fun}")
        else:
            st.write(f"**Minimum Value:** {result.fun}")
        st.write("**At Point:**")
        for i in range(n_vars):
            st.write(f"x{i+1} = {solution[i]:.4f}")
    else:
        st.error("Optimization failed.")
        st.write(result.message)

    # Show KKT conditions
    if show_kkt:
        st.markdown("### Kuhn-Tucker Conditions")

        # Gradient of objective
        grad_f = [diff(f_expr, var) for var in x_vars]

        st.subheader("1. Stationarity")
        for i, g in enumerate(grad_f):
            eq = f"∂f/∂x{i+1} = {g}"
            st.latex(eq)

        st.subheader("2. Primal Feasibility")
        for idx, (ctype, expr) in enumerate(parsed_constraints):
            st.latex(f"g_{idx+1}(x) = {expr} ≤ 0")

        st.subheader("3. Dual Feasibility")
        st.markdown("Lagrange multipliers λᵢ ≥ 0 for inequality constraints")

        st.subheader("4. Complementary Slackness")
        st.markdown("λᵢ * gᵢ(x) = 0")

        st.markdown("**Note:** Actual values of Lagrange multipliers not computed here.")
