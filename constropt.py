import streamlit as st
import numpy as np
from sympy import symbols, sympify, diff, lambdify, Symbol

st.title("Constrained Optimization with Kuhn-Tucker Conditions")

# Variable selection
n_vars = st.slider("Number of decision variables", min_value=1, max_value=6, value=2)
x_vars = symbols(f'x1:{n_vars+1}')  # creates x1 to xn

# Objective function
st.subheader("Objective Function")
obj_type = st.selectbox("Optimization Type", ["Minimize", "Maximize"])
objective_str = st.text_input("Enter objective function (e.g., x1**2 + x2)", "x1**2 + x2")

# Constraints
st.subheader("Constraints (format: expr >= 0 or expr <= 0)")
num_constraints = st.slider("Number of constraints", 0, 6, 2)
constraint_list = []
for i in range(num_constraints):
    constraint_str = st.text_input(f"Constraint {i+1}", f"x1 + x2 - {i+1} <= 0")
    constraint_list.append(constraint_str)

show_kkt = st.checkbox("Show full KKT steps")

if st.button("Solve"):

    # Parse objective
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

    # Parse constraints (convert to g(x) ≥ 0)
    scipy_constraints = []
    parsed_constraints = []

    for con_str in constraint_list:
        try:
            if "<=" in con_str:
                lhs, rhs = con_str.split("<=")
                expr = sympify(rhs) - sympify(lhs)  # convert to ≥ 0
            elif ">=" in con_str:
                lhs, rhs = con_str.split(">=")
                expr = sympify(lhs) - sympify(rhs)  # already ≥ 0
            else:
                st.error("Constraint must contain '<=' or '>='.")
                st.stop()

            func = lambdify([x_vars], expr, modules='numpy')
            scipy_constraints.append({'type': 'ineq', 'fun': func})
            parsed_constraints.append(expr)
        except Exception as e:
            st.error(f"Constraint parse error: {e}")
            st.stop()

    # Run optimizer
    from scipy.optimize import minimize
    try:
        result = minimize(f_func, x0, constraints=scipy_constraints)
    except Exception as e:
        st.error(f"Optimization error: {e}")
        st.stop()

    if result.success:
        st.success("Optimization successful.")
        val = -result.fun if obj_type == "Maximize" else result.fun
        st.write(f"**Optimal value:** {val:.4f}")
        st.write("**Solution:**")
        for i in range(n_vars):
            st.write(f"x{i+1} = {result.x[i]:.4f}")
    else:
        st.error("Optimization failed.")
        st.write(result.message)

    # KKT conditions
    if show_kkt:
        st.subheader("Kuhn-Tucker Conditions")

        # Create Lagrange multipliers λ1, λ2, ...
        lambdas = symbols(f'lambda1:{len(parsed_constraints)+1}')
        L = f_expr
        for lam, g in zip(lambdas, parsed_constraints):
            L -= lam * g

        st.markdown("### Lagrangian")
        st.latex("L(x, \\lambda) = f(x) - \\sum \\lambda_i g_i(x)")
        st.latex(f"L(x, λ) = {L}")

        st.markdown("### 1. Stationarity (∇ₓL = 0)")
        for i, x in enumerate(x_vars):
            dLdx = diff(L, x)
            st.latex(f"\\frac{{\\partial L}}{{\\partial x_{i+1}}} = {dLdx} = 0")

        st.markdown("### 2. Primal Feasibility (constraints ≥ 0)")
        for i, g in enumerate(parsed_constraints):
            st.latex(f"g_{{{i+1}}}(x) = {g} \\geq 0")

        st.markdown("### 3. Dual Feasibility (λ ≥ 0)")
        for i in range(len(parsed_constraints)):
            st.latex(f"\\lambda_{{{i+1}}} \\geq 0")

        st.markdown("### 4. Complementary Slackness")
        for i, g in enumerate(parsed_constraints):
            st.latex(f"\\lambda_{{{i+1}}} \\cdot g_{{{i+1}}}(x) = 0")

        st.info("This version displays symbolic KKT conditions. Solving them requires symbolic/numeric solvers.")
