import streamlit as st
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import sympy as sp

st.title("Constrained Optimization Solver (exact J & H)")

def pretty_round(x, tol=1e-6):
    try:
        if abs(x - round(x)) < tol:
            return int(round(x))
    except Exception:
        pass
    return float(np.round(float(x), 6))

# ---------------- UI ----------------
var_names_str = st.text_input("Variable names (comma-separated)", "wb, wg")
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"])
objective_str = st.text_input("Objective function (use your variables)", "50000 - wb/3 - 2*wg/3")
n_cons = st.number_input("Number of constraints (>= 0)", min_value=0, max_value=20, value=3, step=1)
constraints_str = [
    st.text_input(f"Constraint {i+1} (e.g. x + y - 1 >= 0, or x + y <= 5, or x - y = 2)", 
                  "wb >= 0" if i==0 else ("wg >= 0" if i==1 else "2*sqrt(wg)/3 + sqrt(wb)/3 - 150 >= 0"),
                  key=f"c{i}")
    for i in range(n_cons)
]

# Optional bounds editor (defaults to nonnegativity if left blank)
st.markdown("**Bounds per variable (optional):** leave blank for no bound. Use numbers, e.g. 0 or 10.")
lb_inputs, ub_inputs = [], []
for v in [v.strip() for v in var_names_str.split(",") if v.strip()]:
    c1, c2 = st.columns(2)
    lb_inputs.append(c1.text_input(f"Lower bound for {v}", "0"))
    ub_inputs.append(c2.text_input(f"Upper bound for {v}", ""))

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

    # ---------- Bounds ----------
    def parse_bound(text):
        t = (text or "").strip()
        if t == "": return None
        try:
            return float(t)
        except Exception:
            return None

    lbs = [parse_bound(s) for s in lb_inputs[:n_vars]]
    ubs = [parse_bound(s) for s in ub_inputs[:n_vars]]
    for i in range(n_vars):
        if lb_inputs[i].strip() == "" and ub_inputs[i].strip() == "":
            if lbs[i] is None: lbs[i] = 0.0  # default to nonnegativity

    lb = np.array([(-np.inf if b is None else b) for b in lbs], float)
    ub = np.array([( np.inf if b is None else b) for b in ubs], float)
    bounds = Bounds(lb, ub)

    # ---------- Shared helpers ----------
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
        return sp.simplify(expr)

    # ---------- Objective (value, grad, Hessian) ----------
    obj_expr = parse_and_validate(objective_str, what="objective")
    if max_or_min == "Maximize":
        obj_expr = -obj_expr  # minimize always

    obj_grad_syms = [sp.diff(obj_expr, s) for s in x_syms]
    obj_hess_syms = sp.hessian(obj_expr, x_syms)

    obj_val  = sp.lambdify(x_syms, obj_expr, modules=LAMBDA_MODULES)
    obj_grad = sp.lambdify(x_syms, obj_grad_syms, modules=LAMBDA_MODULES)
    obj_hess = sp.lambdify(x_syms, obj_hess_syms, modules=LAMBDA_MODULES)

    def f(x):  # scalar
        return float(np.asarray(obj_val(*x), float).reshape(()))
    def g(x):  # gradient (n,)
        return np.asarray(obj_grad(*x), float).ravel()
    def H(x):  # Hessian (n,n)
        return np.asarray(obj_hess(*x), float)

    # ---------- Constraints: compile value, gradient, Hessian ----------
    ineq_trips, eq_trips = [], []  # each triplet: (val(x)->float, jac(x)->(n,), hess(x)->(n,n))

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

    def compile_triplet(expr):
        grad = [sp.diff(expr, s) for s in x_syms]
        hess = sp.hessian(expr, x_syms)
        f_  = sp.lambdify(x_syms, expr, modules=LAMBDA_MODULES)
        g_  = sp.lambdify(x_syms, grad, modules=LAMBDA_MODULES)
        H_  = sp.lambdify(x_syms, hess, modules=LAMBDA_MODULES)
        def val(x):
            return float(np.asarray(f_(*x), float).reshape(()))
        def jac(x):
            return np.asarray(g_(*x), float).ravel()
        def hes(x):
            return np.asarray(H_(*x), float)
        return val, jac, hes

    for idx, c_str in enumerate(constraints_str, start=1):
        c_str = c_str.strip()
        if not c_str: continue
        lhs_s, rhs_s, op = split_constraint(c_str)
        lhs = parse_and_validate(lhs_s, what=f"constraint {idx} (lhs)")
        rhs = parse_and_validate(rhs_s, what=f"constraint {idx} (rhs)")
        expr = sp.simplify(lhs - rhs)
        if op == ">=":
            ineq_trips.append(compile_triplet(expr))                 # g(x) >= 0
        elif op == "<=":
            ineq_trips.append(compile_triplet(sp.simplify(rhs - lhs)))  # convert to >= 0
        else:
            eq_trips.append(compile_triplet(expr))                   # h(x) == 0

    # ---------- Initial guess ----------
    def mid_bounds():
        x0 = np.ones(n_vars, dtype=float)
        for i in range(n_vars):
            lo, hi = lb[i], ub[i]
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                x0[i] = 0.5 * (lo + hi)
            elif np.isfinite(lo):
                x0[i] = max(lo + 1e-3, 1.0)
            elif np.isfinite(hi):
                x0[i] = min(hi - 1e-3, 1.0)
        return x0

    x0 = mid_bounds()

    # ---------- SLSQP with EXACT Jacobians for ineq (NEW) ----------
    slsqp_cons = []
    for (val_fun, jac_fun, _) in ineq_trips:
        slsqp_cons.append({"type": "ineq",
                           "fun":  lambda x, f=val_fun:  float(np.asarray(f(x), float).reshape(())),
                           "jac":  lambda x, j=jac_fun:  np.asarray(j(x), float).ravel()})

    try:
        res1 = minimize(
            f, x0, method="SLSQP",
            jac=g,
            bounds=bounds,
            constraints=slsqp_cons,
            options=dict(ftol=1e-12, maxiter=5000, disp=False)  # tighter
        )
    except Exception as e:
        st.error(f"SLSQP failed to start: {e}")
        st.stop()

    # ---------- ALWAYS polish with trust-constr using exact Hessians (NEW) ----------
    nlcs = []

    for (val_fun, jac_fun, hess_fun) in ineq_trips:
        def fun(x, f=val_fun): return float(np.asarray(f(x), float).reshape(()))
        def jac(x, j=jac_fun): return np.asarray(j(x), float).ravel()
        def hess(x, v, Hc=hess_fun):      # v is scalar multiplier passed by trust-constr
            return v * np.asarray(Hc(x), float)
        nlcs.append(NonlinearConstraint(fun, 0.0, np.inf, jac=jac, hess=hess))

    for (val_fun, jac_fun, hess_fun) in eq_trips:
        def fun(x, f=val_fun): return float(np.asarray(f(x), float).reshape(()))
        def jac(x, j=jac_fun): return np.asarray(j(x), float).ravel()
        def hess(x, v, Hc=hess_fun):
            return v * np.asarray(Hc(x), float)
        nlcs.append(NonlinearConstraint(fun, 0.0, 0.0, jac=jac, hess=hess))

    try:
        res2 = minimize(
            f,
            res1.x if np.all(np.isfinite(res1.x)) else x0,
            method="trust-constr",
            jac=g,
            hess=H,                         # exact objective Hessian (NEW)
            bounds=bounds,
            constraints=nlcs,               # each with exact jac & hess
            options=dict(xtol=1e-12, gtol=1e-12, barrier_tol=1e-14, maxiter=20000, verbose=0),
        )
        cand = res2 if (res2.success and res2.fun <= res1.fun) else res1
    except Exception:
        cand = res1

    # ---------- Feasibility check ----------
    def max_violation(x):
        mv = 0.0
        for (val_fun, _, _) in ineq_trips:
            v = float(np.asarray(val_fun(x), float).reshape(()))
            mv = max(mv, max(0.0, -v))
        for (val_fun, _, _) in eq_trips:
            v = abs(float(np.asarray(val_fun(x), float).reshape(())))
            mv = max(mv, v)
        mv = max(mv, float(np.max(np.maximum(0.0, lb - x))))
        mv = max(mv, float(np.max(np.maximum(0.0, x - ub))))
        return mv

    # ---------- Report ----------
    x_star = np.asarray(cand.x, float)
    val = float(f(x_star))
    if max_or_min == "Maximize":
        val = -val

    if getattr(cand, "success", False) and max_violation(x_star) <= 1e-9:
        st.success(
            "Optimal solution found:\n"
            + "\n".join([f"{var_names[i]} = {pretty_round(x_star[i])}" for i in range(n_vars)])
            + f"\nObjective value: {pretty_round(val)}"
        )
    else:
        st.error(f"Optimization not perfectly converged: {getattr(cand, 'message', 'unknown')}")
        st.write("Best iterate found:", {var_names[i]: float(x_star[i]) for i in range(n_vars)})
        st.write("Max constraint/bound residual:", max_violation(x_star))
        st.write("Objective (per your selection):", pretty_round(val))
