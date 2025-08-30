import streamlit as st
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import sympy as sp

st.title("Constrained Optimization Solver (adaptive local, custom variables)")

def pretty_round(x, tol=1e-6):
    try:
        if abs(x - round(x)) < tol:
            return int(round(x))
    except Exception:
        pass
    return float(np.round(float(x), 8))

# ---------------- UI ----------------
var_names_str = st.text_input("Variable names (comma-separated)", "x, y, z")
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"])
objective_str = st.text_input("Objective function (use your variables)", "sqrt(x*y)*z")
n_cons = st.number_input("Number of constraints (>= 0)", min_value=0, max_value=30, value=2, step=1)
constraints_str = [
    st.text_input(f"Constraint {i+1} (e.g. x + y - 1 >= 0, or x + y <= 5, or x - y = 2)", "")
    for i in range(n_cons)
]

with st.expander("Bounds & Solver Settings"):
    st.markdown("**Bounds per variable (optional):** leave blank for no bound. Use numbers, e.g. 0 or 10.")
    lb_inputs, ub_inputs = [], []
    for v in [v.strip() for v in var_names_str.split(",") if v.strip()]:
        c1, c2 = st.columns(2)
        lb_inputs.append(c1.text_input(f"Lower bound for {v}", "0"))
        ub_inputs.append(c2.text_input(f"Upper bound for {v}", ""))

    # Convergence controls
    tol_schedule_str = st.text_input("Tolerance schedule (comma-sep)", "1e-6, 1e-8, 1e-10")
    feas_tol = st.number_input("Feasibility tolerance (reporting)", min_value=1e-12, max_value=1e-4, value=1e-8, step=1e-12, format="%.0e")
    opt_tol  = st.number_input("Optimality tolerance (reporting, ∞-norm grad L approx.)", min_value=1e-12, max_value=1e-4, value=1e-8, step=1e-12, format="%.0e")
    slack_scale = st.number_input("Inequality slack added internally", min_value=0.0, max_value=1e-3, value=0.0, step=1e-12, format="%.0e")
    maxiter_per_pass = st.number_input("Max iterations per pass", min_value=100, max_value=20000, value=4000, step=100)

if st.button("Solve"):
    # ---------- Parse variables ----------
    var_names = [v.strip() for v in var_names_str.split(",") if v.strip()]
    if not var_names:
        st.error("Please provide at least one variable name (e.g., 'x' or 'x, y').")
        st.stop()
    if len(set(var_names)) != len(var_names):
        st.error("Variable names must be unique.")
        st.stop()

    try:
        x_syms = sp.symbols(var_names)
    except Exception as e:
        st.error(f"Could not create symbols from your variable names: {e}")
        st.stop()
    allowed_syms = set(x_syms)
    n_vars = len(var_names)

    # ---------- Parse bounds ----------
    def parse_bound(text):
        t = (text or "").strip()
        if t == "": return None
        try:
            return float(t)
        except Exception:
            return None

    lbs = [parse_bound(s) for s in lb_inputs[:n_vars]] if lb_inputs else [0.0] * n_vars
    ubs = [parse_bound(s) for s in ub_inputs[:n_vars]] if ub_inputs else [None] * n_vars
    # default to nonnegativity if both left blank
    for i in range(n_vars):
        if (not lb_inputs) or (lb_inputs[i].strip() == "" and (not ub_inputs or ub_inputs[i].strip() == "")):
            if lbs[i] is None: lbs[i] = 0.0

    lb = np.array([(-np.inf if b is None else b) for b in lbs], dtype=float)
    ub = np.array([( np.inf if b is None else b) for b in ubs], dtype=float)
    bounds = Bounds(lb, ub)

    # ---------- Helpers ----------
    LAMBDA_MODULES = [{"Abs": np.abs, "Max": np.maximum, "Min": np.minimum}, "numpy"]

    def parse_and_validate(expr_str, what="expression"):
        try:
            expr = sp.sympify(expr_str)
        except Exception as e:
            st.error(f"Error parsing {what}: {e}")
            st.stop()
        extra = expr.free_symbols - allowed_syms
        if extra:
            names = ", ".join(sorted(str(s) for s in extra))
            st.error(f"{what.capitalize()} uses unknown variables: {names}. Add them to 'Variable names'.")
            st.stop()
        return expr

    # ---------- Objective ----------
    obj_expr = parse_and_validate(objective_str, what="objective")
    obj_grad_syms = [sp.diff(obj_expr, s) for s in x_syms]

    obj_func = sp.lambdify(x_syms, obj_expr, modules=LAMBDA_MODULES)
    obj_grad = sp.lambdify(x_syms, obj_grad_syms, modules=LAMBDA_MODULES)

    def obj_for_min(x):
        val = obj_func(*x)
        v = float(np.asarray(val, dtype=float).reshape(()))
        return -v if max_or_min == "Maximize" else v

    def grad_for_min(x):
        g = np.asarray(obj_grad(*x), dtype=float).ravel()
        return -g if max_or_min == "Maximize" else g

    # ---------- Constraints (>=, <=, ==) + scaling ----------
    ineq_exprs, eq_exprs = [], []

    def split_constraint(text):
        if ">=" in text:
            lhs, rhs = text.split(">=", 1); op=">="
        elif "<=" in text:
            lhs, rhs = text.split("<=", 1); op="<="
        elif "==" in text:
            lhs, rhs = text.split("==", 1); op="=="
        elif "=" in text:
            lhs, rhs = text.split("=", 1); op="=="
        else:
            st.error(f"Constraint '{text}' must contain >=, <=, or =.")
            st.stop()
        return lhs.strip(), rhs.strip(), op

    for idx, c_str in enumerate(constraints_str, start=1):
        c_str = c_str.strip()
        if not c_str:
            continue
        lhs_s, rhs_s, op = split_constraint(c_str)
        lhs = parse_and_validate(lhs_s, what=f"constraint {idx} (lhs)")
        rhs = parse_and_validate(rhs_s, what=f"constraint {idx} (rhs)")
        expr = lhs - rhs
        if op == ">=":
            ineq_exprs.append(expr)            # g(x) >= 0
        elif op == "<=":
            ineq_exprs.append(rhs - lhs)       # convert to >= 0
        else:
            eq_exprs.append(expr)              # h(x) == 0

    # Compile and scale constraints so typical magnitude ~ 1 near the start
    def mid_bounds():
        x0 = np.ones(n_vars, dtype=float)
        for i in range(n_vars):
            lo, hi = bounds.lb[i], bounds.ub[i]
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                x0[i] = 0.5 * (lo + hi)
            elif np.isfinite(lo):
                x0[i] = max(lo + 1e-3, 1.0)
            elif np.isfinite(hi):
                x0[i] = min(hi - 1e-3, 1.0)
        return x0

    x0 = mid_bounds()

    def compile_scalar(expr): return sp.lambdify(x_syms, expr, modules=LAMBDA_MODULES)
    def compile_grad(expr):
        grads = [sp.diff(expr, s) for s in x_syms]
        return sp.lambdify(x_syms, grads, modules=LAMBDA_MODULES)

    # raw functions
    ineq_f_raw = [compile_scalar(e) for e in ineq_exprs]
    ineq_g_raw = [compile_grad(e)  for e in ineq_exprs]
    eq_f_raw   = [compile_scalar(e) for e in eq_exprs]
    eq_g_raw   = [compile_grad(e)  for e in eq_exprs]

    # scaling factors based on x0
    def safe_scale(val):
        a = abs(val)
        if not np.isfinite(a) or a < 1e-8: a = 1.0
        return a

    ineq_scale = [safe_scale(float(np.asarray(f(*x0), float).reshape(()))) for f in ineq_f_raw]
    eq_scale   = [safe_scale(float(np.asarray(f(*x0), float).reshape(()))) for f in eq_f_raw]

    # scaled callables
    def ineq_vals(x):
        if not ineq_f_raw: return np.array([], float)
        vals = [(float(np.asarray(f(*x), float).reshape(())) / sc) + float(slack_scale) for f,sc in zip(ineq_f_raw, ineq_scale)]
        return np.array(vals, float)

    def eq_vals(x):
        if not eq_f_raw: return np.array([], float)
        vals = [float(np.asarray(f(*x), float).reshape(())) / sc for f,sc in zip(eq_f_raw, eq_scale)]
        return np.array(vals, float)

    def ineq_grads(x):
        if not ineq_g_raw: return np.empty((0,n_vars), float)
        G = [np.asarray(g(*x), float).ravel() / sc for g,sc in zip(ineq_g_raw, ineq_scale)]
        return np.vstack(G)

    def eq_grads(x):
        if not eq_g_raw: return np.empty((0,n_vars), float)
        G = [np.asarray(g(*x), float).ravel() / sc for g,sc in zip(eq_g_raw, eq_scale)]
        return np.vstack(G)

    # Reporting helpers (unscaled)
    def max_violation_unscaled(x):
        mv = 0.0
        # inequalities (≥0)
        for f in ineq_f_raw:
            v = float(np.asarray(f(*x), float).reshape(()))
            mv = max(mv, max(0.0, -v))
        # equalities (=0)
        for f in eq_f_raw:
            v = abs(float(np.asarray(f(*x), float).reshape(())))
            mv = max(mv, v)
        # bounds
        mv = max(mv, float(np.max(np.maximum(0.0, bounds.lb - x))))
        mv = max(mv, float(np.max(np.maximum(0.0, x - bounds.ub))))
        return mv

    # Rough KKT-ish optimality measure: infinity norm of grad f projected by a least-squares combo of active constraint gradients.
    # (Lightweight and solver-agnostic.)
    def optimality_measure(x):
        g = grad_for_min(x).reshape(-1,1)  # (n,1)
        A = []
        # treat nearly-active scaled ineqs as active (tolerance 1e-6)
        if ineq_f_raw:
            act = np.where(ineq_vals(x) < 1e-6)[0]
            for i in act:
                A.append(ineq_grads(x)[i])
        if eq_f_raw:
            for row in eq_grads(x):
                A.append(row)
        if not A:
            return float(np.linalg.norm(g, ord=np.inf))
        A = np.asarray(A, float)  # (m,n)
        # solve min_{lambda>=0 for ineq approx} ||g + A^T * lambda||_inf (we ignore sign constraints for simplicity)
        lam, *_ = np.linalg.lstsq(A.T, -g, rcond=None)
        r = g + A.T @ lam
        return float(np.linalg.norm(r, ord=np.inf))

    # ---------- Build NonlinearConstraint for trust-constr (scaled) ----------
    nlcs = []
    for i in range(len(ineq_f_raw)):
        def fun(x, idx=i): return ineq_vals(x)[idx]
        def jac(x, idx=i): return ineq_grads(x)[idx]
        nlcs.append(NonlinearConstraint(fun, 0.0, np.inf, jac=jac))
    for i in range(len(eq_f_raw)):
        def fun(x, idx=i): return eq_vals(x)[idx]
        def jac(x, idx=i): return eq_grads(x)[idx]
        nlcs.append(NonlinearConstraint(fun, 0.0, 0.0, jac=jac))

    # ---------- Adaptive tightening loop ----------
    # Parse tolerance schedule
    try:
        tol_schedule = [float(s.strip()) for s in tol_schedule_str.split(",") if s.strip()]
        tol_schedule = [t for t in tol_schedule if t > 0]
    except Exception:
        tol_schedule = [1e-6, 1e-8, 1e-10]
    if not tol_schedule:
        tol_schedule = [1e-6, 1e-8, 1e-10]

    x = x0.copy()
    fval = obj_for_min(x)
    history = []

    for t in tol_schedule:
        res = minimize(
            obj_for_min,
            x,
            method="trust-constr",
            jac=grad_for_min,
            bounds=bounds,
            constraints=nlcs,
            options=dict(xtol=t, gtol=t, barrier_tol=t, maxiter=int(maxiter_per_pass), verbose=0),
        )
        x = np.asarray(res.x, float)
        fval = obj_for_min(x)
        feas = max_violation_unscaled(x)
        optm = optimality_measure(x)
        history.append(dict(tol=t, f=float(fval), feas=float(feas), opt=float(optm), nit=int(res.nit), success=bool(res.success)))

        # Early stop if both measures are already strong
        if feas <= float(feas_tol) and optm <= float(opt_tol):
            break

    # ---------- Report ----------
    st.write("Variables:", var_names)
    st.write("Adaptive passes:")
    for h in history:
        st.write(f"- tol={h['tol']:.1e} | f={pretty_round(h['f'])} | feas_resid={h['feas']:.2e} | opt_resid={h['opt']:.2e} | iters={h['nit']} | success={h['success']}")

    obj_out = -fval if max_or_min == "Maximize" else fval
    feas_final = max_violation_unscaled(x)
    opt_final = optimality_measure(x)

    if (feas_final <= float(feas_tol)) and (opt_final <= float(opt_tol)):
        st.success(
            "Optimal solution (converged):\n"
            + "\n".join([f"{var_names[i]} = {pretty_round(x[i])}" for i in range(n_vars)])
            + f"\nObjective value: {pretty_round(obj_out)}\n"
            + f"Feasibility residual: {feas_final:.2e} (≤ {feas_tol:g})\n"
            + f"Optimality residual:  {opt_final:.2e} (≤ {opt_tol:g})"
        )
    else:
        st.error("Did not meet requested tolerances. Showing tightest-pass result:")
        st.write({var_names[i]: float(x[i]) for i in range(n_vars)})
        st.write(f"Objective: {pretty_round(obj_out)}")
        st.write(f"Feasibility residual: {feas_final:.2e} (target ≤ {feas_tol:g})")
        st.write(f"Optimality residual:  {opt_final:.2e} (target ≤ {opt_tol:g})")
