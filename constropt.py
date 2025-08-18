
import itertools
import math
import streamlit as st
import sympy as sp

st.title("Symbolic Constrained Optimizer (KKT Active-Set, SymPy)")

st.markdown(
    """
This solver attempts a **symbolic** KKT active-set search:
1. Parse your objective \(f(x)\) and inequality constraints \(g_i(x)\ge 0\).
2. Enumerate subsets of constraints assumed active (\(g_i=0\) with multiplier \(\mu_i>0\)).
3. Solve **symbolically** for stationarity and the active equalities.
4. Filter solutions by feasibility and KKT sign conditions (\(\mu_i\ge 0\)).
5. Pick the best feasible candidate.
    """
)

# ----------------------- Inputs -----------------------
n_vars = st.number_input("Number of variables", min_value=1, max_value=6, value=3, step=1)
var_syms = sp.symbols(" ".join([f"x{i+1}" for i in range(n_vars)]), real=True)

sense = st.selectbox("Problem type", ["Maximize", "Minimize"], index=0)
objective_str = st.text_input("Objective f(x1, x2, ...)", "sqrt(x1*x2)*x3")

n_cons = st.number_input("Number of inequality constraints g(x) ≥ 0", min_value=0, max_value=10, value=2, step=1)
con_strs = [st.text_input(f"Constraint {i+1} (e.g. 100 - (x1 + x2) ≥ 0 or lhs ≥ rhs)", "") for i in range(n_cons)]

add_nonneg = st.checkbox("Include nonnegativity constraints xi ≥ 0", value=True)
max_active = st.number_input("Max active constraints to enumerate (keeps search tractable)", min_value=0, max_value=10, value=3, step=1)
limit_solutions_per_subset = st.number_input("Max solutions to keep per subset", min_value=1, max_value=100, value=10, step=1)

use_nsimplify = st.checkbox("Simplify results with nsimplify (exact rationals/radicals when possible)", value=True)

st.markdown("---")

def parse_constraint(s):
    s = s.strip().replace(">=", "≥")
    if "≥" in s:
        lhs, rhs = s.split("≥", 1)
        return sp.simplify(sp.sympify(lhs) - sp.sympify(rhs))
    # If user entered plain expr, assume it's already g(x) ≥ 0 form
    return sp.simplify(sp.sympify(s))

def to_expr(s):
    return sp.simplify(sp.sympify(s))

def feasible_point_check(g_list, subs_dict, tol=sp.Float("1e-10")):
    for g in g_list:
        val = sp.N(g.subs(subs_dict))
        if not val.is_real:
            return False
        if val < -float(tol):
            return False
    return True

def eval_objective(f, subs_dict):
    val = sp.simplify(f.subs(subs_dict))
    try:
        return sp.N(val)
    except Exception:
        return val

def nsimp_optional(x, use_nsimplify=True):
    if not use_nsimplify:
        return x
    try:
        return sp.nsimplify(x, rational=True, maxsteps=100)
    except Exception:
        return x

if st.button("Solve symbolically"):
    # Parse objective
    try:
        f = to_expr(objective_str)
    except Exception as e:
        st.error(f"Error parsing objective: {e}")
        st.stop()

    # Parse constraints g(x) >= 0
    G = []
    labels = []
    for i, s in enumerate(con_strs):
        s = s.strip()
        if not s:
            continue
        try:
            g = parse_constraint(s)
            G.append(g)
            labels.append(f"g{i+1}")
        except Exception as e:
            st.error(f"Error parsing constraint {i+1}: {e}")
            st.stop()

    # Nonnegativity constraints xi >= 0 as extra g's
    if add_nonneg:
        for i, xi in enumerate(var_syms):
            G.append(xi)  # xi >= 0
            labels.append(f"x{i+1}≥0")

    nG = len(G)
    st.write(f"Total inequality constraints considered: {nG}")

    # Precompute gradients
    grad_f = [sp.diff(f, xi) for xi in var_syms]
    grad_g = [[sp.diff(g, xi) for xi in var_syms] for g in G]

    best = None
    candidates_shown = 0

    # Enumerate active sets (by size up to max_active)
    for k in range(0, min(nG, int(max_active)) + 1):
        for active_idx_tuple in itertools.combinations(range(nG), k):
            active_idx = set(active_idx_tuple)

            # Build stationarity equations: grad f - sum(mu_i * grad g_i) = 0 (for active i only)
            mus = sp.symbols(" ".join([f"mu{i}" for i in active_idx]), nonnegative=True)  # μ_i ≥ 0
            mu_map = dict(zip(active_idx, mus))

            stationarity_eqs = []
            for j in range(n_vars):
                expr = grad_f[j]
                for i_idx in active_idx:
                    expr -= mu_map[i_idx] * grad_g[i_idx][j]
                stationarity_eqs.append(sp.Eq(sp.simplify(expr), 0))

            # Active constraints equality: g_i = 0
            active_eqs = [sp.Eq(sp.simplify(G[i_idx]), 0) for i_idx in active_idx]

            # Unknowns are x's and the μ's
            unknowns = list(var_syms) + list(mus)

            # Try to solve symbolically
            sols = []
            try:
                sol = sp.solve(stationarity_eqs + active_eqs, unknowns, dict=True, simplify=True)
                if isinstance(sol, dict):
                    sols = [sol]
                else:
                    sols = sol
            except Exception:
                sols = []

            # Clip number of solutions per subset
            if len(sols) > int(limit_solutions_per_subset):
                sols = sols[: int(limit_solutions_per_subset)]

            # Check solutions: feasibility (all g >= 0) and μ_i >= 0
            for sol in sols:
                # Reject parametric solutions
                if any(isinstance(v, sp.Symbol) for v in sol.values()):
                    continue

                # Extract x values
                x_subs = {xi: sp.simplify(sol.get(xi, sp.nan)) for xi in var_syms}
                # Reject non-real or NaN
                bad = False
                for xi in var_syms:
                    val = x_subs[xi]
                    if val.has(sp.I) or val is sp.nan:
                        bad = True
                        break
                if bad:
                    continue

                # Evaluate μ's and ensure nonnegativity
                mus_ok = True
                for i_idx, mu_sym in mu_map.items():
                    mu_val = sp.N(sol.get(mu_sym, sp.nan))
                    if mu_val is sp.nan or (not mu_val.is_real) or mu_val < -1e-12:
                        mus_ok = False
                        break
                if not mus_ok:
                    continue

                # Feasibility of all constraints
                if not feasible_point_check(G, x_subs):
                    continue

                # Evaluate objective
                f_val = eval_objective(f, x_subs)

                cand = {
                    "x_subs": x_subs,
                    "active": [labels[i] for i in sorted(active_idx)],
                    "mus": {labels[i]: sp.simplify(sol[mu_map[i]]) for i in sorted(active_idx)} if mus else {},
                    "f_val": f_val,
                }

                # Keep best by sense
                if best is None:
                    best = cand
                else:
                    if sense == "Maximize":
                        if sp.N(cand["f_val"]) > sp.N(best["f_val"]):
                            best = cand
                    else:
                        if sp.N(cand["f_val"]) < sp.N(best["f_val"]):
                            best = cand

                # Show a few candidates for transparency
                if candidates_shown < 25:
                    st.markdown("**Candidate found:**")
                    st.write("Active set:", cand["active"])
                    st.write("Multipliers μ:", {k: nsimp_optional(v, use_nsimplify) for k, v in cand["mus"].items()})
                    st.write("x:", {str(k): nsimp_optional(v, use_nsimplify) for k, v in cand["x_subs"].items()})
                    st.write("f(x):", nsimp_optional(cand["f_val"], use_nsimplify))
                    st.markdown("---")
                    candidates_shown += 1

    if best is None:
        st.warning("No feasible KKT point found in the enumerated active sets. "
                   "Consider increasing 'Max active constraints' or revising the model. "
                   "Note: Highly non-polynomial objectives/constraints may require broader algebraic methods.")
    else:
        st.success("Best KKT candidate (symbolic):")
        st.write("Active set:", best["active"] if best["active"] else "(none)")
        st.write("x* =", {str(k): nsimp_optional(v, use_nsimplify) for k, v in best["x_subs"].items()})
        st.write("f(x*) =", nsimp_optional(best["f_val"], use_nsimplify))
        st.caption("Note: This method searches necessary KKT points. For non-convex problems, global optimality is not guaranteed without additional checks.")
