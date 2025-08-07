import streamlit as st
import sympy as sp
from itertools import product

st.set_page_config(page_title="KKT Solver Robust", layout="wide")
st.title("Robust Kuhn-Tucker (KKT) Solver with Case Enumeration")

opt_type = st.selectbox("Choose optimization type", ["Maximize", "Minimize"])
n_vars = st.slider("Number of variables (up to 4)", 1, 4, 2)

# Define vars with positive=True (important for log/sqrt domain)
vars = sp.symbols(f"x1:{n_vars+1}", real=True, positive=True)
var_names = [str(v) for v in vars]

obj_str = st.text_input(f"Objective function f({', '.join(var_names)})", "log(x1) + log(x2)")

n_constraints = st.slider("Number of constraints (up to 8)", 1, 8, 3)
constraint_strs = []
for i in range(n_constraints):
    default = f"{var_names[i % n_vars]} >= 0"
    constraint_strs.append(st.text_input(f"Constraint #{i+1} (≥ 0)", default))

solve_button = st.button("Solve KKT")

if solve_button:
    try:
        f = sp.sympify(obj_str)
    except Exception:
        st.error("Invalid objective function.")
        st.stop()

    g_exprs = []
    for cs in constraint_strs:
        try:
            if ">=" in cs:
                left, right = map(str.strip, cs.split(">="))
                g = sp.sympify(f"({left}) - ({right})")
            else:
                g = sp.sympify(cs)
            g_exprs.append(g)
        except Exception:
            st.error(f"Invalid constraint: {cs}")
            st.stop()

    # Multipliers
    lambdas = sp.symbols(f"l0:{len(g_exprs)}", real=True, nonnegative=True)

    L = f
    for lam, g in zip(lambdas, g_exprs):
        L += lam * g

    dL_dx = [sp.diff(L, v) for v in vars]

    # Complementary slackness cases: each multiplier zero or >0
    # We'll represent >0 by None and 0 explicitly.
    cases = list(product([0, None], repeat=len(g_exprs)))

    solutions = []

    st.markdown(f"### Trying {len(cases)} KKT complementary slackness cases...")

    for case_i, case in enumerate(cases, 1):
        eqs = []
        subs = {}

        # Setup complementary slackness:
        # lambda_i = 0 or g_i = 0
        for i, val in enumerate(case):
            if val == 0:
                subs[lambdas[i]] = 0
            else:
                eqs.append(sp.Eq(g_exprs[i], 0))

        # Add first order conditions, substitute zeros
        for eq in dL_dx:
            eqs.append(sp.Eq(eq.subs(subs), 0))

        unknowns = list(vars)
        unknowns += [lambdas[i] for i, val in enumerate(case) if val is None]

        try:
            sols = sp.solve(eqs, unknowns, dict=True)
        except Exception as e:
            st.write(f"Case {case_i}: Solve failed with error {e}")
            continue

        if not sols:
            st.write(f"Case {case_i}: No solution.")
            continue

        for sol in sols:
            # Insert known multipliers
            sol.update(subs)

            # Validate solution
            valid = True

            # Check all vars positive
            for v in vars:
                val = sol.get(v, None)
                if val is None or val.is_real is False or val.is_finite is False:
                    valid = False
                    break
                try:
                    if val.evalf() <= 0:
                        valid = False
                        break
                except Exception:
                    valid = False
                    break
            if not valid:
                continue

            # Check constraints g_i >= 0
            for g in g_exprs:
                try:
                    val = g.subs(sol).evalf()
                    if val < -1e-8:
                        valid = False
                        break
                except Exception:
                    valid = False
                    break
            if not valid:
                continue

            # Check multipliers nonnegative
            for lam in lambdas:
                val = sol.get(lam, 0)
                if val.is_real is False or val.is_finite is False:
                    valid = False
                    break
                try:
                    if val.evalf() < -1e-8:
                        valid = False
                        break
                except Exception:
                    valid = False
                    break
            if not valid:
                continue

            # Check complementary slackness
            for i, lam in enumerate(lambdas):
                gval = g_exprs[i].subs(sol).evalf()
                lamval = sol.get(lam, 0).evalf()
                if abs(lamval * gval) > 1e-6:
                    valid = False
                    break
            if not valid:
                continue

            # Evaluate objective
            try:
                val = f.subs(sol).evalf()
                if not val.is_real or not val.is_finite:
                    continue
            except Exception:
                continue

            solutions.append((sol, val, case_i))

    if not solutions:
        st.error("❌ No feasible KKT solution found.")
    else:
        best_sol = max(solutions, key=lambda x: x[1]) if opt_type == "Maximize" else min(solutions, key=lambda x: x[1])
        st.success(f"Found {len(solutions)} feasible solutions.")

        for sol, val, case_i in solutions:
            st.markdown(f"#### Case #{case_i} solution:")
            items = []
            for k in sorted(sol.keys(), key=lambda x: str(x)):
                items.append(f"${sp.latex(k)} = {sp.latex(sol[k])}$")
            st.latex(",\quad ".join(items))
            st.latex(f"f = {val}")

        st.markdown("### 🏆 Optimal Solution:")
        sol, val, case_i = best_sol
        items = []
        for k in sorted(sol.keys(), key=lambda x: str(x)):
            items.append(f"${sp.latex(k)} = {sp.latex(sol[k])}$")
        st.latex(",\quad ".join(items))
        st.latex(f"\\textbf{{Optimal value}}: f = {val}")
