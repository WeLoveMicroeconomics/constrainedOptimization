import streamlit as st
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import sympy as sp

st.title("Constrained Optimization Solver (scaled & polished)")

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

    lb = np.array([(-np.inf if b is None else b) for b in lbs], dtype=float)
    ub = np.array([( np.inf if b is None else b) for b in ubs], dtype=float)
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
        return expr

    # ---------- Objective ----------
    obj_expr = parse_and_validate(objective_str, what="objective")
    obj_grad_syms = [sp.diff(obj_expr, s) for s in x_syms]

    try:
        obj_func = sp.lambdify(x_syms, obj_expr, modules=LAMBDA_MODULES)
        obj_grad = sp.lambdify(x_syms, obj_grad_syms, modules=LAMBDA_MODULES)
    except Exception as e:
        st.error(f"Error compiling objective: {e}")
        st.stop()

    def obj_value(x):
        val = obj_func(*x)
        return float(np.asarray(val, dtype=float).reshape(()))

    def obj_grad_x(x):
        g = np.asarray(obj_grad(*x), dtype=float).ravel()
        return g

    def obj_for_min(x):
        v = obj_value(x)
        return -v if max_or_min == "Maximize" else v

    def grad_for_min(x):
        g = obj_grad_x(x)
        return -g if max_or_min == "Maximize" else g

    # ---------- Constraints parsing (>=, <=, = supported) ----------
    ineq_exprs = []  # g(x) >= 0
    eq_exprs   = []  # h(x) = 0

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
            ineq_exprs.append(expr)
        elif op == "<=":
            ineq_exprs.append(sp.expand(rhs - lhs))  # convert to >= 0
        else:
            eq_exprs.append(expr)

    # Compile scalar & gradients
    def compile_scalar(expr): return sp.lambdify(x_syms, expr, modules=LAMBDA_MODULES)
    def compile_grad(expr):
        grads = [sp.diff(expr, s) for s in x_syms]
        return sp.lambdify(x_syms, grads, modules=LAMBDA_MODULES)

    ineq_f = [compile_scalar(e) for e in ineq_exprs]
    ineq_g = [compile_grad(e)  for e in ineq_exprs]
    eq_f   = [compile_scalar(e) for e in eq_exprs]
    eq_g   = [compile_grad(e)  for e in eq_exprs]

    # ---------- NEW: choose a scaled space y = x / s ----------
    # scale each variable to be O(1): use finite width if both bounds exist, else use max(1, |bound|, |mid|)
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
    # variable scales
    s = np.ones(n_vars, dtype=float)
    for i in range(n_vars):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]) and ub[i] > lb[i]:
            s[i] = max(1.0, (ub[i] - lb[i]) / 2.0)
        else:
            s[i] = max(1.0, abs(x0[i]), abs(lb[i]) if np.isfinite(lb[i]) else 0.0, abs(ub[i]) if np.isfinite(ub[i]) else 0.0)

    # mapping
    def to_x(y): return s * y
    def to_y(x): return x / s

    # scaled bounds
    lb_y = lb / s
    ub_y = ub / s
    bounds_y = Bounds(lb_y, ub_y)

    # ---------- NEW: objective scaling so magnitudes ~1 ----------
    obj0 = obj_value(x0)
    obj_scale = max(1.0, abs(obj0))
    if not np.isfinite(obj_scale): obj_scale = 1.0

    def f_y(y):
        return obj_for_min(to_x(y)) / obj_scale

    def g_y(y):
        # chain rule: df/dy = s * df/dx
        g = grad_for_min(to_x(y))
        return (g * s) / obj_scale

    # ---------- NEW: constraints in y-space with scaled gradients ----------
    def make_ineq_fun(j):
        f = ineq_f[j]; g = ineq_g[j]
        def fun(y):
            val = f(*to_x(y))
            return float(np.asarray(val, float).reshape(()))
        def jac(y):
            gx = np.asarray(g(*to_x(y)), float).ravel()
            return gx * s
        return fun, jac

    def make_eq_fun(j):
        f = eq_f[j]; g = eq_g[j]
        def fun(y):
            val = f(*to_x(y))
            return float(np.asarray(val, float).reshape(()))
        def jac(y):
            gx = np.asarray(g(*to_x(y)), float).ravel()
            return gx * s
        return fun, jac

    # for SLSQP we only pass ineq constraints
    slsqp_cons = []
    for j in range(len(ineq_f)):
        fun_j, _ = make_ineq_fun(j)
        slsqp_cons.append({"type": "ineq", "fun": fun_j})

    # ---------- Feasibility score (in y-space) ----------
    def max_violation_y(y):
        mv = 0.0
        for j in range(len(ineq_f)):
            v = make_ineq_fun(j)[0](y)
            mv = max(mv, max(0.0, -v))
        for j in range(len(eq_f)):
            v = abs(make_eq_fun(j)[0](y))
            mv = max(mv, v)
        mv = max(mv, float(np.max(np.maximum(0.0, lb_y - y))))
        mv = max(mv, float(np.max(np.maximum(0.0, y - ub_y))))
        return mv

    # ---------- Initial guess in y-space ----------
    y0 = to_y(x0)

    st.write("Variables:", var_names)
    st.write("Scale factors (x = s * y):", {var_names[i]: float(s[i]) for i in range(n_vars)})
    st.write("Start (y-space):", {var_names[i]: float(y0[i]) for i in range(n_vars)})

    # ---------- Phase 1: SLSQP (fast) on scaled problem ----------
    try:
        res1 = minimize(
            f_y,
            y0,
            method="SLSQP",
            jac=g_y,
            bounds=bounds_y,
            constraints=slsqp_cons,
            options=dict(ftol=1e-12, maxiter=5000, disp=False),  # CHANGED: tighter & more iters
        )
    except Exception as e:
        st.error(f"SciPy failed to start SLSQP: {e}")
        st.stop()

    y_cand = np.asarray(res1.x, float)

    # ---------- Phase 2: trust-constr polish with exact constraint J ---------- 
    # Build NonlinearConstraints in y-space
    nlcs = []
    for j in range(len(ineq_f)):
        fun_j, jac_j = make_ineq_fun(j)
        nlcs.append(NonlinearConstraint(fun_j, 0.0, np.inf, jac=jac_j))
    for j in range(len(eq_f)):
        fun_j, jac_j = make_eq_fun(j)
        nlcs.append(NonlinearConstraint(fun_j, 0.0, 0.0, jac=jac_j))

    try:
        res2 = minimize(
            f_y,
            y_cand,
            method="trust-constr",
            jac=g_y,
            bounds=bounds_y,
            constraints=nlcs,
            options=dict(xtol=1e-12, gtol=1e-12, barrier_tol=1e-12, maxiter=10000, verbose=0),  # CHANGED: much tighter
        )
    except Exception as e:
        res2 = res1  # fallback to SLSQP result

    # pick the best (scaled objective already comparable)
    cand = res2 if (getattr(res2, "success", False) and res2.fun <= res1.fun) else res1

    # ---------- Report ----------
    x_sol = to_x(cand.x)
    val_scaled = f_y(cand.x)
    val = val_scaled * obj_scale
    if max_or_min == "Maximize":
        val = -val

    viol = max_violation_y(cand.x)

    if getattr(cand, "success", False) and viol <= 1e-9:
        st.success(
            "Optimal solution found:\n"
            + "\n".join([f"{var_names[i]} = {pretty_round(x_sol[i])}" for i in range(n_vars)])
            + f"\nObjective value: {pretty_round(val)}"
        )
        st.caption(f"Max residual (unscaled constraints & bounds) ≈ {viol:.2e}")
    else:
        st.error(f"Optimization not perfectly converged: {getattr(cand, 'message', 'unknown')}")
        st.write("Best iterate found:", {var_names[i]: float(x_sol[i]) for i in range(n_vars)})
        st.write("Max residual:", viol)
        st.write("Objective (per your selection):", pretty_round(val))
