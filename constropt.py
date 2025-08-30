import streamlit as st
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import sympy as sp

st.title("Constrained Optimization Solver (robust, custom variables)")

def pretty_round(x, tol=1e-6):
    try:
        if abs(x - round(x)) < tol:
            return int(round(x))
    except Exception:
        pass
    return float(np.round(float(x), 6))

# ---------------- UI ----------------
var_names_str = st.text_input("Variable names (comma-separated)", "x, y, z")
max_or_min = st.selectbox("Problem type", ["Minimize", "Maximize"])
objective_str = st.text_input("Objective function (use your variables)", "sqrt(x*y)*z")
n_cons = st.number_input("Number of constraints (>= 0)", min_value=0, max_value=20, value=2, step=1)
constraints_str = [
    st.text_input(f"Constraint {i+1} (e.g. x + y - 1 >= 0, or x + y <= 5, or x - y = 2)", "")
    for i in range(n_cons)
]

# Optional bounds editor (defaults to nonnegativity if left blank)
st.markdown("**Bounds per variable (optional):** leave blank for no bound. Use numbers, e.g. 0 or 10.")
lb_inputs = []
ub_inputs = []
for v in [v.strip() for v in var_names_str.split(",") if v.strip()]:
    cols = st.columns(2)
    lb_inputs.append(cols[0].text_input(f"Lower bound for {v}", "0"))
    ub_inputs.append(cols[1].text_input(f"Upper bound for {v}", ""))

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

    lbs = [parse_bound(s) for s in lb_inputs[:n_vars]]
    ubs = [parse_bound(s) for s in ub_inputs[:n_vars]]
    # default to nonnegativity if completely unspecified in the UI row
    for i in range(n_vars):
        if lb_inputs[i].strip() == "" and ub_inputs[i].strip() == "":
            if lbs[i] is None: lbs[i] = 0.0
            # ub stays None

    bounds = Bounds(
        [(-np.inf if b is None else b) for b in lbs],
        [( np.inf if b is None else b) for b in ubs],
    )

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
        return expr

    # ---------- Objective ----------
    obj_expr = parse_and_validate(objective_str, what="objective")
    # Sympy gradients for trust-constr
    obj_grad_syms = [sp.diff(obj_expr, s) for s in x_syms]

    try:
        obj_func = sp.lambdify(x_syms, obj_expr, modules=LAMBDA_MODULES)
        obj_grad = sp.lambdify(x_syms, obj_grad_syms, modules=LAMBDA_MODULES)
    except Exception as e:
        st.error(f"Error compiling objective: {e}")
        st.stop()

    def obj_for_min(x):
        val = obj_func(*x)
        v = float(np.asarray(val, dtype=float).reshape(()))
        return -v if max_or_min == "Maximize" else v

    def grad_for_min(x):
        g = np.asarray(obj_grad(*x), dtype=float).ravel()
        return -g if max_or_min == "Maximize" else g

    # ---------- Constraints parsing (>=, <=, = supported) ----------
    # We convert each constraint to one or two numerical functions:
    #   - for >= : g(x) = lhs - rhs  (want g >= 0)
    #   - for <= : g(x) = rhs - lhs  (want g >= 0)
    #   - for  = : h(x) = lhs - rhs  (want h = 0)
    ineq_funcs = []    # list of callables g(x) >= 0
    ineq_grads = []    # their gradients
    eq_funcs = []      # list of h(x) == 0
    eq_grads = []      # their gradients

    def compile_scalar(expr):
        return sp.lambdify(x_syms, expr, modules=LAMBDA_MODULES)

    def compile_grad(expr):
        grads = [sp.diff(expr, s) for s in x_syms]
        return sp.lambdify(x_syms, grads, modules=LAMBDA_MODULES)

    def add_ineq(expr):
        ineq_funcs.append(compile_scalar(expr))
        ineq_grads.append(compile_grad(expr))

    def add_eq(expr):
        eq_funcs.append(compile_scalar(expr))
        eq_grads.append(compile_grad(expr))

    # Accept operators in order of specificity
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
            add_ineq(expr)
        elif op == "<=":
            add_ineq(rhs - lhs)  # convert to >= 0
        else:  # equality
            add_eq(expr)

    # Numeric slack to avoid borderline rejections
    SLACK = 1e-9

    # SLSQP-style constraint wrappers (ineq only; equalities handled via trust-constr)
    slsqp_cons = []
    for f in ineq_funcs:
        def cons_fun(x, f=f):
            val = f(*x)
            return float(np.asarray(val, dtype=float).reshape(())) + SLACK
        slsqp_cons.append({"type": "ineq", "fun": cons_fun})

    # ---------- Feasibility finder (COBYLA on total violation) ----------
    def total_violation(x):
        tv = 0.0
        # inequalities: want g(x) >= 0
        for f in ineq_funcs:
            v = float(np.asarray(f(*x), dtype=float).reshape(()))
            tv += max(0.0, -(v + SLACK))
        # equalities: penalize absolute residual
        for f in eq_funcs:
            v = float(np.asarray(f(*x), dtype=float).reshape(()))
            tv += abs(v)
        # bounds as soft constraints
        for i in range(n_vars):
            if np.isfinite(bounds.lb[i]) and x[i] < bounds.lb[i]:
                tv += bounds.lb[i] - x[i]
            if np.isfinite(bounds.ub[i]) and x[i] > bounds.ub[i]:
                tv += x[i] - bounds.ub[i]
        return tv

    # initial guess from mid-bounds or 1.0
    def initial_guess():
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

    x0 = initial_guess()

    # Use COBYLA to reduce violation (it only supports ineq; we fold eq into |h| <= s trick via penalties inside objective)
    try:
        # Convert bounds to COBYLA inequalities
        cobyla_cons = []
        for i in range(n_vars):
            if np.isfinite(bounds.lb[i]):
                cobyla_cons.append({"type":"ineq", "fun": (lambda j=i: (lambda x: x[j] - bounds.lb[j]))()})
            if np.isfinite(bounds.ub[i]):
                cobyla_cons.append({"type":"ineq", "fun": (lambda j=i: (lambda x: bounds.ub[j] - x[j]))()})
        # Convert user ineqs to COBYLA
        for f in ineq_funcs:
            cobyla_cons.append({"type":"ineq", "fun": (lambda g=f: (lambda x: float(np.asarray(g(*x), float).reshape(())) + SLACK))()})
        res_feas = minimize(lambda x: total_violation(x), x0, method="COBYLA", constraints=cobyla_cons,
                            options=dict(maxiter=600, rhobeg=1.0, tol=1e-6, disp=False))
        if res_feas.success and total_violation(res_feas.x) < 1e-6:
            x0 = res_feas.x
    except Exception:
        pass  # fall back to heuristic x0

    # Diagnostics
    st.write("Variables:", var_names)
    st.write("Start point:", {var_names[i]: float(x0[i]) for i in range(n_vars)})
    st.write("Total violation at start (≈0 is good):", total_violation(x0))

    # ---------- Solve with SLSQP, then fall back to trust-constr ----------
    options = dict(ftol=1e-9, maxiter=1200, disp=False)
    try:
        res = minimize(
            obj_for_min,
            x0,
            method="SLSQP",
            jac=grad_for_min,          # gradient helps
            bounds=bounds,
            constraints=slsqp_cons,    # only inequalities here
            options=options,
        )
    except Exception as e:
        st.error(f"SciPy failed to start SLSQP: {e}")
        st.stop()

    # Check result feasibility
    def max_violation(x):
        mv = 0.0
        for f in ineq_funcs:
            v = float(np.asarray(f(*x), dtype=float).reshape(())) + SLACK
            mv = max(mv, max(0.0, -v))
        for f in eq_funcs:
            v = abs(float(np.asarray(f(*x), dtype=float).reshape(())))
            mv = max(mv, v)
        # bounds
        for i in range(n_vars):
            if np.isfinite(bounds.lb[i]): mv = max(mv, max(0.0, bounds.lb[i] - x[i]))
            if np.isfinite(bounds.ub[i]): mv = max(mv, max(0.0, x[i] - bounds.ub[i]))
        return mv

    need_fallback = (not res.success) or (max_violation(res.x) > 1e-6)

    if need_fallback:
        # Build NonlinearConstraints for trust-constr (supports equalities)
        nlcs = []
        for f, g in zip(ineq_funcs, ineq_grads):
            def fun(x, f=f): return float(np.asarray(f(*x), float).reshape(())) + SLACK
            def jac(x, g=g): return np.asarray(g(*x), float).ravel()
            nlcs.append(NonlinearConstraint(fun, 0.0, np.inf, jac=jac))
        for f, g in zip(eq_funcs, eq_grads):
            def fun(x, f=f): return float(np.asarray(f(*x), float).reshape(()))
            def jac(x, g=g): return np.asarray(g(*x), float).ravel()
            nlcs.append(NonlinearConstraint(fun, 0.0, 0.0, jac=jac))

        try:
            res2 = minimize(
                obj_for_min,
                res.x if np.all(np.isfinite(res.x)) else x0,
                method="trust-constr",
                jac=grad_for_min,
                bounds=bounds,
                constraints=nlcs,
                options=dict(xtol=1e-9, gtol=1e-9, barrier_tol=1e-9, maxiter=2000, verbose=0),
            )
            # Use the better of the two
            cand = res2 if (res2.success and res2.fun <= res.fun) or not res.success else res
        except Exception:
            cand = res  # keep SLSQP result if trust-constr errors out
    else:
        cand = res

    # ---------- Report ----------
    if cand.success and max_violation(cand.x) <= 1e-6:
        sol = cand.x
        val = obj_for_min(sol)
        if max_or_min == "Maximize":
            val = -val
        st.success(
            "Optimal solution found:\n"
            + "\n".join([f"{var_names[i]} = {pretty_round(sol[i])}" for i in range(n_vars)])
            + f"\nObjective value: {pretty_round(val)}"
        )
    else:
        st.error(f"Optimization failed: {cand.message}")
        st.write("Best iterate found:", {var_names[i]: float(cand.x[i]) for i in range(n_vars)})
        st.write("Max constraint violation:", max_violation(cand.x))
        try:
            val = obj_for_min(cand.x)
            st.write("Objective (per your selection):", -val if max_or_min == "Maximize" else val)
        except Exception:
            pass
