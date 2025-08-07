import streamlit as st
import sympy as sp
from itertools import product

st.set_page_config(page_title="KKT Solver", layout="wide")
st.title("Kuhn-Tucker (KKT) Constrained Optimization Solver")

# Step 1: User inputs
opt_type = st.selectbox("Choose optimization type", ["Maximize", "Minimize"])
n_vars = st.slider("Number of choice variables", 1, 4, 2)

# Define variables
vars = sp.symbols(f"x1:{n_vars+1}", real=True)
var_names = [str(v) for v in vars]

# Objective function
obj_str = st.text_input(f"Objective function f({', '.join(var_names)})", "log(x1) + log(x2)")

# Constraints
constraint_count = st.slider("Number of constraints (up to 8)", 1, 8, 3)
constraint_strs = []
for i in range(constraint_count):
    default = f"{var_names[i % n_vars]} >= 0"
    constraint_strs.append(st.text_input(f"Constraint #{i+1} (≥ 0)", default))

submit = st.button("Solve")

if submit:
    st.markdown("## Step 1: Setup")

    try:
        f = sp.sympify(obj_str)
    except:
        st.error("Invalid objective function.")
        st.stop()

    # Parse constraints
    g_exprs = []
    for con_str in constraint_strs:
        try:
            if ">=" in con_str:
                left, right = map(str.strip, con_str.split(">="))
                g = sp.sympify(f"({left}) - ({right})")
            else:
                g = sp.sympify(con_str)
            g_exprs.append(g)
        except:
            st.error(f"Invalid constraint: {con_str}")
            st.stop()

    # Lagrange multipliers
    lambdas = sp.symbols(f"l0:{len(g_exprs)}", real=True, nonnegative=True)

    # Build Lagrangian
    L = f
    for lam, g in zip(lambdas, g_exprs):
        L += lam * g

    focs = [sp.Eq(sp.diff(L, v), 0) for v in vars]
    slack_conditions = [sp.Eq(lambdas[i] * g_exprs[i], 0) for i in range(len(g_exprs))]

    st.markdown("### First-Order Conditions")
    for eq in focs:
        st.latex(sp.latex(eq))

    st.markdown("### Complementary Slackness Conditions")
    for i, cond in enumerate(slack_conditions):
        g_latex = sp.latex(g_exprs[i])
        st.latex(f"\\lambda_{{{i+1}}} \\geq 0,\\ g_{{{i+1}}}(x) = {g_latex} \\geq 0,\\ \\lambda_{{{i+1}}} \\cdot g_{{{i+1}}}(x) = 0")

    st.markdown("## Step 2: Solving KKT Cases")
    active_sets = list(product([0, 1], repeat=len(g_exprs)))
    solutions = []

    for case_index, activity in enumerate(active_sets):
        eqs = focs.copy()

        for i, act in enumerate(activity):
            if act == 0:
                eqs.append(sp.Eq(lambdas[i], 0))
            else:
                eqs.append(sp.Eq(g_exprs[i], 0))

        all_syms = list(vars) + list(lambdas)

        try:
            sol_set = sp.nonlinsolve(eqs, tuple(all_syms))
            for sol in sol_set:
                sol_dict = dict(zip(all_syms, sol))

                # Discard complex or infinite solutions
                if any((not v.is_real or not v.is_finite) for v in sol_dict.values()):
                    continue

                # Check feasibility of constraints
                feasibility = all([g.subs(sol_dict).evalf() >= -1e-6 for g in g_exprs])

                # Also check domain of objective function
                try:
                    val = f.subs(sol_dict).evalf()
                    if feasibility and val.is_real and val.is_finite:
                        solutions.append((sol_dict, val, case_index))
                except Exception:
                    continue
        except Exception:
            pass

    if not solutions:
        st.error("❌ No feasible KKT solution found.")
    else:
        extrema_func = max if opt_type == "Maximize" else min
        best_val = None
        best_sol = None

        for sol, val, idx in solutions:
            st.markdown(f"### ✅ Feasible KKT Case #{idx + 1}")
            latex_sol = ",\_
