"""v3 joint batter-bowler pressure model: the SINGLE estimation stage for player usual ability and pressure effects.

Rows and likelihoods (sign convention: every player effect is positive = good for that player):
  S  legal ball, immediate batter runs r          r ~ N(eta_S, phi_S * xvar)        (quasi-likelihood, variance from the
        eta_S = xruns + Xs.b + a_s + b_s f_bat - g_s - d_s f_bowl + m_s               player-blind outcome distribution)
  W  legal ball, bowler-credited dismissal of the striker   d ~ Bernoulli(sigmoid(eta_W))
        eta_W = logit(p0) + Xw.b - a_w - b_w f_bat + g_w + d_w f_bowl + m_w           (run-outs: in neither player's likelihood)
  X  every delivery, extra runs from a wide / no-ball (bowler only)   x ~ N(eta_X, phi_X * vx_d),  eta_X = p_x cx + c - e
  batter (a_s,b_s,a_w,b_w) ~ N(0, Sig_bat) 4x4; bowler (g_s,d_s,g_w,d_w,e) ~ N(0, Sig_bowl) 5x5; match (m_s,m_w) ~ N(0, diag)
  fixed: competition-season levels (S and W), population pressure slopes for f_bat and f_bowl (S and W), extras intercept.
Estimation: Laplace-approximate restricted ML by EM (E-step = Newton MAP + inverse Hessian, M-step = closed-form covariance
updates), iterated to convergence (no fixed iteration cap below 2000). Player posteriors = N(MAP, inverse-Hessian block).
Run-value of a player at state s (batting-side runs, relative to an average player in the same state):
  bat : a_s + b_s f(s) - VW(s) [sigmoid(l(s) - a_w - b_w f(s)) - sigmoid(l(s))]
  bowl: g_s + d_s f(s) + VW(s) [sigmoid(l(s) + g_w + d_w f(s)) - sigmoid(l(s))] + e * (1 + p_x/(1-p_x))
where VW(s) = expected future innings runs after a dot ball minus after a wicket (both AFTER the ball), so the ball's own runs
are only in the immediate-scoring term."""
import numpy as np, pandas as pd, scipy.sparse as sp, torch, time, os
torch.set_num_threads(int(os.environ.get("RR_THREADS", "16")))

BAT_K, BOW_K, MAT_K = 4, 5, 2

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def sig(x): return 1 / (1 + np.exp(-x))

def fshape(u, kind):
    if kind == "sq": return u ** 2 - 1 / 3
    if kind == "lin": return u - 0.5
    if kind == "bin": return (u >= 0.8).astype(float) - 0.2
    raise ValueError(kind)

class Design:
    """Builds the sparse design for a delivery table L (legal rows) and A (all rows)."""
    def __init__(self, L, A, fkind="sq", slopes=True, pop_slope=True, groups="season", match=True, players=True, min_balls=30, wL=None, wA=None):
        self.fkind = fkind
        fb = fshape(L.u_bat.values, fkind) if slopes or pop_slope else np.zeros(len(L))
        fw = fshape(L.u_bowl.values, fkind) if slopes or pop_slope else np.zeros(len(L))
        # players with < min_balls deliveries share ONE pooled effect per side (id -1): they can never be eligible and their own
        # effects are not identifiable; pooling keeps their deliveries in the likelihood without a separate parameter each
        nbb = L.striker_id.value_counts(); nww = A.bowler_id.value_counts()
        sb = L.striker_id.where(L.striker_id.map(nbb) >= min_balls, -1); sw_ = A.bowler_id.where(A.bowler_id.map(nww) >= min_balls, -1)
        swL = L.bowler_id.where(L.bowler_id.map(nww) >= min_balls, -1)
        self.bats = pd.Index(np.sort(sb.unique())); self.bows = pd.Index(np.sort(sw_.unique()))
        self.mats = pd.Index(np.sort(A.match_id.unique()))
        gkey = (L.family + "_" + L.season_year.astype(str)) if groups == "season" else L.family
        self.groups = pd.Index(np.sort(gkey.unique()))
        G = len(self.groups); gi = self.groups.get_indexer(gkey)
        # fixed columns: S groups [0,G), S f_bat, S f_bowl, W groups, W f_bat, W f_bowl, X intercept
        ps = G + 2; self.p = p = 2 * ps + 1
        nb, nw, nm = len(self.bats), len(self.bows), len(self.mats)
        self.ob = p; self.ow = p + BAT_K * nb; self.om = self.ow + BOW_K * nw; self.dim = self.om + MAT_K * nm
        bi = self.bats.get_indexer(sb); wi = self.bows.get_indexer(swL); mi = self.mats.get_indexer(L.match_id)
        wiA = self.bows.get_indexer(sw_)
        n = len(L); nA = len(A)
        ps_ = 1.0 if pop_slope else 0.0; pl = 1.0 if players else 0.0; sl = 1.0 if (slopes and players) else 0.0; mt = 1.0 if match else 0.0
        # S rows: 8 entries each
        cS = np.column_stack([gi, np.full(n, G), np.full(n, G + 1), self.ob + BAT_K * bi, self.ob + BAT_K * bi + 1,
                              self.ow + BOW_K * wi, self.ow + BOW_K * wi + 1, self.om + MAT_K * mi])
        vS = np.column_stack([np.ones(n), ps_ * fb, ps_ * fw, pl * np.ones(n), sl * fb, -pl * np.ones(n), -sl * fw, mt * np.ones(n)])
        cW = np.column_stack([ps + gi, np.full(n, ps + G), np.full(n, ps + G + 1), self.ob + BAT_K * bi + 2, self.ob + BAT_K * bi + 3,
                              self.ow + BOW_K * wi + 2, self.ow + BOW_K * wi + 3, self.om + MAT_K * mi + 1])
        vW = np.column_stack([np.ones(n), ps_ * fb, ps_ * fw, -pl * np.ones(n), -sl * fb, pl * np.ones(n), sl * fw, mt * np.ones(n)])
        cX = np.column_stack([np.full(nA, 2 * ps), self.ow + BOW_K * wiA + 4]); vX = np.column_stack([np.ones(nA), -pl * np.ones(nA)])
        self.cS, self.vS, self.cW, self.vW, self.cX, self.vX = cS, vS, cW, vW, cX, vX
        self.AS = sp.csr_matrix((vS.ravel(), (np.repeat(np.arange(n), 8), cS.ravel())), shape=(n, self.dim))
        self.AW = sp.csr_matrix((vW.ravel(), (np.repeat(np.arange(n), 8), cW.ravel())), shape=(n, self.dim))
        self.AX = sp.csr_matrix((vX.ravel(), (np.repeat(np.arange(nA), 2), cX.ravel())), shape=(nA, self.dim))
        self.yS = L.bat_runs.values.astype(float); self.oS = L.xruns.values; self.vS0 = L.xvar.values
        self.dW = L.bowler_wkt.values.astype(float)
        pl_ = L.p_bwkt.values / np.maximum(1 - L.p_x.values, 1e-6)          # P(bowler wicket | legal ball)
        self.oW = logit(pl_)
        il = (A.is_wide | A.is_nb).values
        self.yX = np.where(il, A.bowler_runs.values, 0.0).astype(float); self.oX = A.p_x.values * A.cx.values
        self.vX0 = np.maximum(A.p_x.values * (A.vx.values + A.cx.values ** 2) - (A.p_x.values * A.cx.values) ** 2, 1e-3)
        self.flags = dict(slopes=slopes, pop_slope=pop_slope, match=match, players=players)
        # leverage weights: a ball counts in proportion to how much it can cost the player's side (default: all 1)
        self.wW = np.ones(n) if wL is None else np.asarray(wL, float)
        if wL is not None: self.vS0 = self.vS0 / self.wW
        if wA is not None: self.vX0 = self.vX0 / np.asarray(wA, float)
        # unused random columns (switched-off components) get a tiny prior variance so they stay at zero
        self.n = n; self.nA = nA

def _pack(Sb, Sw, sm, phiS, phiX): return np.concatenate([Sb.ravel(), Sw.ravel(), sm, [phiS, phiX]])
def _unpack(v):
    a = BAT_K * BAT_K; b = a + BOW_K * BOW_K
    return v[:a].reshape(BAT_K, BAT_K), v[a:b].reshape(BOW_K, BOW_K), v[b:b + MAT_K], v[b + MAT_K], v[b + MAT_K + 1]

def _psd(S, floor):
    S = (S + S.T) / 2; w, Q = np.linalg.eigh(S); return (Q * np.maximum(w, floor)) @ Q.T

DEVICE = torch.device(os.environ.get("RR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

class _Fitter:
    """Dense linear algebra (Hessian, Cholesky, inverse, M-step traces) runs on DEVICE (GPU when available); sparse design
    products stay on the CPU. Identical operations on either device: float64 Hessian, Jacobi scaling, float32 factorisation,
    float64 residuals."""
    def __init__(self, D, floor, dtype=torch.float32, device=None):
        self.D = D; self.floor = floor; self.dtype = dtype; self.dev = device or DEVICE
        self.nb, self.nw, self.nm = len(D.bats), len(D.bows), len(D.mats)
        self.ASt, self.AWt, self.AXt = D.AS.T.tocsr(), D.AW.T.tocsr(), D.AX.T.tocsr()
        self.HS = torch.from_numpy((self.ASt @ sp.diags(1 / D.vS0) @ D.AS).toarray()).to(self.dev)   # constant Gaussian information (x 1/phi)
        self.HX = torch.from_numpy((self.AXt @ sp.diags(1 / D.vX0) @ D.AX).toarray()).to(self.dev)
        self.fl = D.flags
        cn = np.asarray(abs(D.AS).sum(0) + abs(D.AW).sum(0) + abs(D.AX).sum(0)).ravel()[:D.p]
        self.zero_fixed = np.where(cn == 0)[0]                  # switched-off fixed columns are pinned at 0
        self.ib = torch.as_tensor(D.ob + np.arange(self.nb) * BAT_K, device=self.dev); self.iw = torch.as_tensor(D.ow + np.arange(self.nw) * BOW_K, device=self.dev)
        self.im = torch.as_tensor(D.om + np.arange(self.nm) * MAT_K, device=self.dev)
    def mask(self, v):
        Sb, Sw, sm, phiS, phiX = _unpack(v.copy()); fl = self.fl; eps = 1e-12
        if not fl["slopes"] or not fl["players"]:
            for S_ in (Sb, Sw):
                for k in (1, 3): S_[k, :] = 0; S_[:, k] = 0; S_[k, k] = eps
        if not fl["players"]: Sb = np.diag(np.full(BAT_K, eps)); Sw = np.diag(np.full(BOW_K, eps))
        if not fl["match"]: sm = np.full(MAT_K, eps)
        keep = lambda S_: np.where(np.diag(S_) <= eps * 10, 1, 0)
        Sb2 = _psd(Sb, self.floor); Sw2 = _psd(Sw, self.floor)
        for S_, S2 in ((Sb, Sb2), (Sw, Sw2)):             # switched-off components stay switched off
            off = np.diag(S_) <= eps * 10
            S2[off, :] = 0; S2[:, off] = 0; S2[off, off] = eps
        sm = np.where(sm <= eps * 10, eps, np.maximum(sm, self.floor))
        return _pack(Sb2, Sw2, sm, max(phiS, 1e-3), max(phiX, 1e-3))
    def penalty(self, Pb, Pw, Pm):
        D = self.D; rows = []; cols = []; vals = []
        for o, k, K, P in ((D.ob, self.nb, BAT_K, Pb), (D.ow, self.nw, BOW_K, Pw)):
            base = o + np.arange(k) * K
            for i in range(K):
                for j in range(K): rows.append(base + i); cols.append(base + j); vals.append(np.full(k, P[i, j]))
        base = D.om + np.arange(self.nm) * MAT_K
        for k in range(MAT_K): rows.append(base + k); cols.append(base + k); vals.append(np.full(self.nm, Pm[k]))
        rows.append(self.zero_fixed); cols.append(self.zero_fixed); vals.append(np.ones(len(self.zero_fixed)))
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    def step(self, v, th, newton_tol=1e-8):
        """One EM step from parameters v: MAP (Newton), Laplace LL at v, M-step -> new v."""
        D = self.D; nb, nw, nm = self.nb, self.nw, self.nm; dev = self.dev
        Sb, Sw, sm, phiS, phiX = _unpack(v)
        Pb = np.linalg.inv(Sb); Pw = np.linalg.inv(Sw); Pm = 1 / sm
        pr, pc, pv = self.penalty(Pb, Pw, Pm)
        pr_t = torch.as_tensor(pr, device=dev); pc_t = torch.as_tensor(pc, device=dev); pv_t = torch.as_tensor(pv, device=dev)
        def pen_mv(t):
            g = np.zeros(D.dim)
            g[D.ob:D.ow] = (t[D.ob:D.ow].reshape(nb, BAT_K) @ Pb).ravel()
            g[D.ow:D.om] = (t[D.ow:D.om].reshape(nw, BOW_K) @ Pw).ravel()
            g[D.om:] = (t[D.om:].reshape(nm, MAT_K) * Pm).ravel(); return g
        wS = 1 / (phiS * D.vS0); wX = 1 / (phiX * D.vX0)
        def grad(th):
            eS = D.oS + D.AS @ th; eW = D.oW + D.AW @ th; eX = D.oX + D.AX @ th; mu = sig(eW)
            return -(self.ASt @ (wS * (D.yS - eS))) + self.AWt @ (D.wW * (mu - D.dW)) - (self.AXt @ (wX * (D.yX - eX))) + pen_mv(th), mu
        nt = 0
        for outer in range(30):
            g, mu = grad(th)
            H = self.HS / phiS + self.HX / phiX + torch.from_numpy((self.AWt @ sp.diags(D.wW * mu * (1 - mu)) @ D.AW).toarray()).to(dev)
            H[pr_t, pc_t] += pv_t
            dsc = torch.sqrt(torch.diagonal(H)); Hs = (H / dsc[:, None] / dsc[None, :]).to(self.dtype)   # Jacobi scaling
            del H
            Lc = torch.linalg.cholesky(Hs); del Hs; nt += 1
            done = False
            for inner in range(25):                          # chord iterations re-using the factorisation
                rhs = (torch.from_numpy(-g).to(dev) / dsc).to(self.dtype)[:, None]
                st = (torch.cholesky_solve(rhs, Lc)[:, 0].double() / dsc).cpu().numpy()
                th = th + st
                if np.max(np.abs(st)) < newton_tol: done = True; break
                g, mu = grad(th)
            if done and inner < 3: break
        g, mu = grad(th)
        C = torch.cholesky_inverse(Lc).double() / dsc[:, None] / dsc[None, :]
        logdetH = 2 * float(torch.log(torch.diagonal(Lc).double()).sum()) + 2 * float(torch.log(dsc).sum())
        del Lc
        eS = D.oS + D.AS @ th; eW = D.oW + D.AW @ th; eX = D.oX + D.AX @ th
        rS = D.yS - eS; rX = D.yX - eX
        nll = 0.5 * np.sum(wS * rS ** 2) + 0.5 * np.sum(np.log(2 * np.pi * phiS * D.vS0)) + np.sum(D.wW * (np.logaddexp(0, eW) - D.dW * eW))             + 0.5 * np.sum(wX * rX ** 2) + 0.5 * np.sum(np.log(2 * np.pi * phiX * D.vX0))
        ldG = nb * np.linalg.slogdet(Sb)[1] + nw * np.linalg.slogdet(Sw)[1] + nm * np.sum(np.log(sm))
        ll = -(nll + 0.5 * th @ pen_mv(th)) - 0.5 * ldG - 0.5 * logdetH
        def blockcov(o, k, K, idx):
            u = th[o:o + K * k].reshape(k, K)
            Cb = torch.stack([C[idx + i, idx + j] for i in range(K) for j in range(K)], dim=-1).reshape(k, K, K).cpu().numpy()
            return (u.T @ u + Cb.sum(0)) / k, u, Cb
        Sb2, ub, Cbb = blockcov(D.ob, nb, BAT_K, self.ib); Sw2, uw, Cbw = blockcov(D.ow, nw, BOW_K, self.iw)
        um = th[D.om:].reshape(nm, MAT_K)
        sm2 = np.array([(um[:, k] ** 2 + C[self.im + k, self.im + k].cpu().numpy()).mean() for k in range(MAT_K)])
        phiS2 = (np.sum(rS ** 2 / D.vS0) + float((C * self.HS).sum())) / D.n   # tr(W A C A') = sum(C * A'WA)
        phiX2 = (np.sum(rX ** 2 / D.vX0) + float((C * self.HX).sum())) / D.nA
        Cdiag = torch.diagonal(C).cpu().numpy(); del C
        extra = dict(th=th, C=Cdiag, ub=ub, uw=uw, Cbb=Cbb, Cbw=Cbw, ll=ll, grad_inf=float(np.max(np.abs(g))), newton=nt)
        return self.mask(_pack(Sb2, Sw2, sm2, phiS2, phiX2)), extra

def _scale(v):
    Sb, Sw, sm, phiS, phiX = _unpack(v)
    sb = np.sqrt(np.outer(np.diag(Sb), np.diag(Sb))); sw = np.sqrt(np.outer(np.diag(Sw), np.diag(Sw)))
    return _pack(sb, sw, sm, phiS, phiX)

def fit(D, init=None, tol=1e-4, max_em=2000, verbose=False, floor=1e-7, dtype=torch.float32, flat_window=10**9, device=None):
    """Laplace-EM with SQUAREM acceleration. Converged when the largest change of a plain EM step, relative to the parameter's
    scale (covariances relative to sqrt(var_i var_j)), is below tol AND the Laplace LL changes by < 1e-3."""
    Fz = _Fitter(D, floor, dtype, device)
    max_em = int(os.environ.get("RR_MAXEM", max_em))          # debugging only; never set in the final run (manifest records env)
    if init is None:
        v = _pack(np.diag([0.05 ** 2, 0.03 ** 2, 0.1 ** 2, 0.05 ** 2]), np.diag([0.05 ** 2, 0.03 ** 2, 0.1 ** 2, 0.05 ** 2, 0.02 ** 2]),
                  np.array([0.05 ** 2, 0.05 ** 2]), 1.0, 1.0); th = np.zeros(D.dim)
    else:
        v = _pack(*[np.array(init[k], dtype=float) for k in ("Sb", "Sw", "sm", "phiS", "phiX")])
        th = init["theta"].copy() if init.get("theta") is not None and len(init["theta"]) == D.dim else np.zeros(D.dim)
    v = Fz.mask(v); trace = []; t0 = time.time(); conv = False; nstep = 0; crit = None
    v1, ex = Fz.step(v, th); nstep += 1
    while nstep < max_em:
        rel = float(np.max(np.abs(v1 - v) / (_scale(v) + 1e-12)))
        Sb_, Sw_, sm_, _, _ = _unpack(v)
        sds = np.round(np.sqrt(np.concatenate([np.diag(Sb_), np.diag(Sw_), sm_])), 5).tolist()
        imax = int(np.argmax(np.abs(v1 - v) / (_scale(v) + 1e-12)))
        trace.append(dict(step=nstep, ll=ex["ll"], rel_change=rel, worst_param=imax, newton=ex["newton"], grad_inf=ex["grad_inf"], sec=round(time.time() - t0, 1), sds=sds))
        if verbose: print({k: x for k, x in trace[-1].items() if k != "sds" or len(trace) % 10 == 1}, flush=True)
        if len(trace) > 2 and rel < tol and abs(trace[-1]["ll"] - trace[-2]["ll"]) < 1e-3: conv = True; crit = "parameter_change"; break
        # flat-likelihood rule: remaining movement is a component creeping along a flat ridge / towards its boundary
        if len(trace) > flat_window + 1 and trace[-1]["ll"] - trace[-1 - flat_window]["ll"] < 0.1 and rel < 50 * tol: conv = True; crit = "flat_likelihood"; break
        v2, ex2 = Fz.step(v1, ex["th"]); nstep += 1
        r = v1 - v; w = v2 - v1 - r
        if np.linalg.norm(w) > 0:
            a = min(-np.linalg.norm(r) / np.linalg.norm(w), -1.0)
            vx = Fz.mask(v - 2 * a * r + a * a * w)
            v3, ex3 = Fz.step(vx, ex2["th"]); nstep += 1
            if ex3["ll"] >= ex2["ll"] - 1e-6: v, v1, ex = vx, v3, ex3
            else: v, v1, ex = v1, v2, ex2
        else: v, v1, ex = v1, v2, ex2
    Sb, Sw, sm, phiS, phiX = _unpack(v)
    return dict(theta=ex["th"], Cdiag=ex["C"], Sb=Sb, Sw=Sw, sm=sm, phiS=phiS, phiX=phiX, ll=ex["ll"], trace=trace, converged=conv,
                em_iters=nstep, criterion=crit, ub=ex["ub"], uw=ex["uw"], Cbb=ex["Cbb"], Cbw=ex["Cbw"], design=D, grad_inf=ex["grad_inf"], fitter=Fz, v=v)

def init_from(F):
    return dict(Sb=F["Sb"], Sw=F["Sw"], sm=F["sm"], phiS=F["phiS"], phiX=F["phiX"], theta=F["theta"])

# ---------------- run value over a reference distribution ----------------
def run_value(F, Q, side, draws=None, ids=None):
    """Mean run value of each player over reference states Q (DataFrame with f_bat/f_bowl, logit l0, VW, px).
    draws: optional (k, ndraw, K) array of effect draws; returns (k,) or (k, ndraw)."""
    D = F["design"]
    if side == "bat":
        U = F["ub"] if draws is None else draws          # (k,K) or (k,nd,K)
        f = Q.f_bat.values; l = Q.l0.values; vw = Q.VW.values; p0 = sig(l)
        if U.ndim == 2: U = U[:, None, :]
        lin = U[..., 0] + U[..., 1] * f.mean()
        wk = np.zeros(U.shape[:2])
        for j in range(0, len(f), 500):
            fj, lj, vj, pj = f[j:j + 500], l[j:j + 500], vw[j:j + 500], p0[j:j + 500]
            pi = sig(lj[None, None, :] - U[..., 2:3] - U[..., 3:4] * fj[None, None, :])
            wk += (vj * (pi - pj)).sum(-1)
        out = lin - wk / len(f)
    else:
        U = F["uw"] if draws is None else draws
        f = Q.f_bowl.values; l = Q.l0.values; vw = Q.VW.values; p0 = sig(l); xf = 1 + (Q.px.values / (1 - Q.px.values)).mean()
        if U.ndim == 2: U = U[:, None, :]
        lin = U[..., 0] + U[..., 1] * f.mean() + U[..., 4] * xf
        wk = np.zeros(U.shape[:2])
        for j in range(0, len(f), 500):
            fj, lj, vj, pj = f[j:j + 500], l[j:j + 500], vw[j:j + 500], p0[j:j + 500]
            pi = sig(lj[None, None, :] + U[..., 2:3] + U[..., 3:4] * fj[None, None, :])
            wk += (vj * (pi - pj)).sum(-1)
        out = lin + wk / len(f)
    return out[:, 0] if draws is None else out

def draw_effects(F, side, nd, rng):
    u = F["ub"] if side == "bat" else F["uw"]; C = F["Cbb"] if side == "bat" else F["Cbw"]
    K = u.shape[1]; Lc = np.linalg.cholesky(C + 1e-12 * np.eye(K))
    z = rng.standard_normal((u.shape[0], nd, K))
    return u[:, None, :] + np.einsum("kij,knj->kni", Lc, z)
