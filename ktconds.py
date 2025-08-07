import streamlit as st
import sympy as sp
from itertools import product

st.title("Kuhn-Tucker Optimization Solver")

# Step 1: User inputs
opt_type = st.selectbox("Choose optimization type", ["Maximize", "Minimize"])
n_vars = st.slider("Number of choice variables (x1, x2, ..., xn)", 1, 4, 2)

vars = sp.symbols(" ".join([f"x{i+1}" for i in range(n_vars)]))
var_names = [str(v) for v in vars]

obj_str = st.text_input(f"Objective function f({', '.join(var_names)})", "x1 + x2")
constraint_count = st.slider("Number of constraints (including x1≥0, ...)", 1, 8, 3)

constraint_strs = []
for i in range(constraint_count):
    default = f"{var_names[i % n_vars]} >= 0"
    constraint_strs.append(st.text_input(f"Constraint #{i+1} (≥ 0)", default))

submit = st.button("Solve")

if submit:
    st.markdown("## Results")

    # Step 2: Symbolic setup
    f = sp.sympify(obj_str)
    constraints = []
    g_exprs = []
    for con_str in constraint_strs:
        if ">=" in con_str:
            left, right = map(str.strip, con_str.split(">="))
            g = sp.sympify(f"({left}) - ({right})")
        else:
            g = sp.sympify(con_str)
        g_exprs.append(g)
        constraints.append(g >= 0)

    lambdas = sp.symbols(f"l0:{len(g_exprs)}", real=True, nonnegative=True)

    # Build Lagrangian
    L = f
    for lam, g in zip(lambdas, g_exprs):
        L += lam * g

    # First-order conditions
    focs = [sp.Eq(sp.diff(L, v), 0) for v in vars]

    # Complementary slackness conditions
    slack_conditions = [sp.Eq(lambdas[i] * g_exprs[i], 0) for i in range(len(g_exprs))]

    # All KKT cases: active (λ>0) or inactive (λ=0)
    active_sets = list(product([0, 1], repeat=len(g_exprs)))  # 0: inactive, 1: active

    solutions = []
    for case_index, activity in enumerate(active_sets):
        eqs = focs.copy()
        assumptions = []

        # Build system for this KKT case
        for i, act in enumerate(activity):
            if act == 0:
                eqs.append(sp.Eq(lambdas[i], 0))
            else:
                eqs.append(sp.Eq(g_exprs[i], 0))

        try:
            sol = sp.solve(eqs, (*vars, *lambdas), dict=True)
            if sol:
                for s in sol:
                    feasible = all([g.subs(s).evalf() >= -1e-6 for g in g_exprs])
                    if feasible:
                        val = f.subs(s).evalf()
                        solutions.append((s, val, case_index))
        except Exception:
            pass

    if not solutions:
        st.latex("No\\ feasible\\ solution\\ found.")
    else:
        # Display all feasible solutions
        best_val = None
        best_sol = None
        extrema_func = max if opt_type == "Maximize" else min

        for sol, val, idx in solutions:
            st.markdown(f"### Feasible KKT Case #{idx + 1}")
            latex_sol = ",\\ ".join([f"{str(k)} = {sp.latex(v)}" for k, v in sol.items()])
            st.latex(f"Solution:\\ {latex_sol}")
            st.latex(f"{'f' if opt_type=='Maximize' else 'f'} = {val}")
            if best_val is None or extrema_func(val, best_val) == val:
                best_val = val
                best_sol = sol

        st.markdown("## Optimal Solution")
        latex_sol = ",\\ ".join([f"{str(k)} = {sp.latex(v)}" for k, v in best_sol.items()])
        st.latex(f"\\textbf{{Optimal}}:\\ {latex_sol}")
        st.latex(f"\\textbf{{Optimal value}}:\\ f = {best_val}")

    # Show first order and slackness conditions
    st.markdown("## First-Order Conditions (Symbolic)")
    for eq in focs:
        st.latex(sp.latex(eq))

    st.markdown("## Complementary Slackness Conditions")
    for i, cond in enumerate(slack_conditions):
        st.latex(f"\\lambda_{i+1} \\geq 0,\\ g_{i+1}(x) \\geq 0,\\ {sp.latex(cond)}")

