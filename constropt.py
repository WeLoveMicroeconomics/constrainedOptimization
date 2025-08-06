import streamlit as st
import sympy as sp

st.title("KKT Conditions Solver (Step-by-step)")

# Step 1: User inputs
n_vars = st.slider("Number of decision variables", 1, 3, 2)
n_cons = st.slider("Number of inequality constraints", 0, 3, 1)

x_vars = sp.symbols(f'x1:{n_vars+1}')
lambdas = sp.symbols(f'lambda1:{n_cons+1}')

obj_str = st.text_input("Objective function (e.g. x1**2 + x2)", "x1**2 + x2")
constraint_strs = [st.text_input(f"Constraint {i+1} (e.g. x1 + x2 - 1 <= 0)", f"x1 + x2 - {i+1} <= 0") for i in range(n_cons)]

if st.button("Solve using KKT"):
    try:
        f = sp.sympify(obj_str)
    except Exception as e:
        st.error(f"Invalid objective function: {e}")
        st.stop()

    g_list = []
    for s in constraint_strs:
        if "<=" in s:
            lhs, rhs = s.split("<=")
            g = sp.sympify(rhs) - sp.sympify(lhs)
        elif ">=" in s:
            lhs, rhs = s.split(">=")
            g = sp.sympify(lhs) - sp.sympify(rhs)
        else:
            st.error("Constraints must include <= or >=")
            st.stop()
        g_list.append(g)

    # Step 2: Construct the Lagrangian
    L = f - sum(lambdas[i]*g_list[i] for i in range(n_cons))
    st.markdown("### Lagrangian")
    st.latex(f"L(x, \\lambda) = {sp.latex(L)}")

    # Step 3: Compute gradients
    st.markdown("### Stationarity Conditions")
    stationarity = []
    for i, x in enumerate(x_vars):
        dLdx = sp.diff(L, x)
        stationarity.append(dLdx)
        st.latex(f"\\frac{{\\partial L}}{{\\partial x_{i+1}}} = {sp.latex(dLdx)} = 0")

    # Step 4: Complementary slackness
    st.markdown("### Complementary Slackness")
    slackness = []
    for i, g in enumerate(g_list):
        cs = lambdas[i] * g
        slackness.append(cs)
        st.latex(f"\\lambda_{{{i+1}}} \\cdot g_{{{i+1}}}(x) = {sp.latex(cs)} = 0")

    # Step 5: Dual feasibility
    st.markdown("### Dual Feasibility")
    for i in range(n_cons):
        st.latex(f"\\lambda_{{{i+1}}} \geq 0")

    # Step 6: Solve the system
    st.markdown("### Solving the KKT System")
    equations = stationarity + slackness + g_list  # primal feasibility implicit in g >= 0
    sol = sp.solve(equations, x_vars + lambdas, dict=True, real=True)

    if not sol:
        st.warning("No KKT points found.")
        st.stop()

    feasible = []
    st.markdown("### KKT Candidates")
    for s in sol:
        point = {v: s[v] for v in x_vars}
        g_vals = [g.subs(s).evalf() for g in g_list]
        lambda_vals = [s.get(l, 0) for l in lambdas]
        feasible_flag = all(gv >= -1e-6 for gv in g_vals) and all(lv >= -1e-6 for lv in lambda_vals)
        obj_val = f.subs(s).evalf()
        st.write(f"Point: {point}, Objective: {obj_val:.4f}, Feasible: {feasible_flag}")
        if feasible_flag:
            feasible.append((point, obj_val))

    if feasible:
        best = min(feasible, key=lambda t: t[1])
        st.success(f"Optimal Solution: {best[0]} with objective value {best[1]:.4f}")
    else:
        st.warning("No feasible KKT points found.")
