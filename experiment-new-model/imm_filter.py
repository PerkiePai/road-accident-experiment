"""
IMM (Interacting Multiple Models) filter — CV + CA modes.

Maintains two parallel Kalman filters:
  - CV  (Constant Velocity):     state = [x, z, vx, vz]
  - CA  (Constant Acceleration): state = [x, z, vx, vz, ax, az]

Each frame, mixing weights are updated via likelihood ratio so the CA
model automatically gains weight during sudden deceleration (collision)
and CV dominates during normal straight-line driving.

Usage:
    f = IMMFilter(dt=1/30)
    f.init(x0, z0)
    for each frame:
        f.predict()
        speed_kmh = f.update(x_m, z_m)   # returns smoothed speed in km/h
        mu_cv, mu_ca = f.weights          # mixing weights (sum to 1)
"""

import numpy as np


class IMMFilter:
    def __init__(self, dt: float = 1 / 30.0,
                 q_cv: float = 0.5,
                 q_ca: float = 1.0,
                 r: float = 0.3,
                 p_cv_cv: float = 0.95,
                 p_ca_ca: float = 0.95):
        """
        dt      : seconds between frames
        q_cv    : process noise std for CV model (m/s²)
        q_ca    : process noise std for CA model (m/s³ — jerk)
        r       : measurement noise std (metres)
        p_cv_cv : probability of staying in CV mode
        p_ca_ca : probability of staying in CA mode
        """
        self.dt = dt
        self.r  = r

        # Markov transition matrix [from_mode, to_mode]
        self.Pi = np.array([[p_cv_cv,       1 - p_cv_cv],
                            [1 - p_ca_ca,   p_ca_ca    ]])

        # ── CV model (4-state: x z vx vz) ─────────────────────
        self.F_cv = np.eye(4)
        self.F_cv[0, 2] = dt
        self.F_cv[1, 3] = dt

        # Process noise (discrete Wiener): acceleration noise
        q = q_cv
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q_cv = q**2 * np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ])

        # ── CA model (6-state: x z vx vz ax az) ───────────────
        self.F_ca = np.eye(6)
        self.F_ca[0, 2] = dt; self.F_ca[0, 4] = 0.5 * dt2
        self.F_ca[1, 3] = dt; self.F_ca[1, 5] = 0.5 * dt2
        self.F_ca[2, 4] = dt
        self.F_ca[3, 5] = dt

        # Process noise (jerk-driven)
        q3 = q_ca
        self.Q_ca = q3**2 * np.array([
            [dt4*dt2/36, 0,          dt3*dt2/12, 0,          dt2*dt2/6, 0         ],
            [0,          dt4*dt2/36, 0,          dt3*dt2/12, 0,         dt2*dt2/6 ],
            [dt3*dt2/12, 0,          dt4/4,      0,          dt3/2,     0         ],
            [0,          dt3*dt2/12, 0,          dt4/4,      0,         dt3/2     ],
            [dt2*dt2/6,  0,          dt3/2,      0,          dt2,       0         ],
            [0,          dt2*dt2/6,  0,          dt3/2,      0,         dt2       ],
        ])

        # Measurement matrix (observe x and z only)
        self.H_cv = np.zeros((2, 4)); self.H_cv[0, 0] = 1; self.H_cv[1, 1] = 1
        self.H_ca = np.zeros((2, 6)); self.H_ca[0, 0] = 1; self.H_ca[1, 1] = 1
        self.R    = r**2 * np.eye(2)

        self._initialized = False

    def init(self, x0: float, z0: float):
        self.x_cv = np.array([x0, z0, 0.0, 0.0])
        self.P_cv = np.diag([1.0, 1.0, 4.0, 4.0])

        self.x_ca = np.array([x0, z0, 0.0, 0.0, 0.0, 0.0])
        self.P_ca = np.diag([1.0, 1.0, 4.0, 4.0, 2.0, 2.0])

        self.mu = np.array([0.8, 0.2])   # start mostly CV
        self._initialized = True

    @property
    def weights(self):
        return self.mu.copy()

    def _likelihood(self, innov, S):
        n    = innov.shape[0]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e-300
        maha2 = float(innov @ np.linalg.solve(S, innov))
        return np.exp(-0.5 * (maha2 + logdet + n * np.log(2 * np.pi)))

    def update(self, x_m: float, z_m: float) -> float:
        """Feed one measurement, return smoothed speed in km/h."""
        if not self._initialized:
            self.init(x_m, z_m)
            return 0.0

        z_meas = np.array([x_m, z_m])

        # ── Step 1: mixing (interaction) ───────────────────────
        c_j   = self.Pi.T @ self.mu          # predicted mode probs
        mu_ij = (self.Pi * self.mu[:, None]) / c_j[None, :]   # (2,2)

        # Mixed CV initial conditions
        x0_cv = mu_ij[0, 0] * self.x_cv + mu_ij[1, 0] * self.x_ca[:4]
        P0_cv = (mu_ij[0, 0] * (self.P_cv +
                  np.outer(self.x_cv - x0_cv, self.x_cv - x0_cv)) +
                 mu_ij[1, 0] * (self.P_ca[:4, :4] +
                  np.outer(self.x_ca[:4] - x0_cv, self.x_ca[:4] - x0_cv)))

        x_ca_4 = self.x_ca[:4]
        x0_ca4 = mu_ij[0, 1] * self.x_cv + mu_ij[1, 1] * x_ca_4
        P_ca4  = self.P_ca[:4, :4]
        P0_ca4 = (mu_ij[0, 1] * (self.P_cv +
                   np.outer(self.x_cv - x0_ca4, self.x_cv - x0_ca4)) +
                  mu_ij[1, 1] * (P_ca4 +
                   np.outer(x_ca_4 - x0_ca4, x_ca_4 - x0_ca4)))
        x0_ca = np.concatenate([x0_ca4, self.x_ca[4:]])
        P0_ca = self.P_ca.copy()
        P0_ca[:4, :4] = P0_ca4

        # ── Step 2: predict ────────────────────────────────────
        xp_cv = self.F_cv @ x0_cv
        Pp_cv = self.F_cv @ P0_cv @ self.F_cv.T + self.Q_cv

        xp_ca = self.F_ca @ x0_ca
        Pp_ca = self.F_ca @ P0_ca @ self.F_ca.T + self.Q_ca

        # ── Step 3: update (Kalman correction) ────────────────
        def kf_update(xp, Pp, H, R):
            innov = z_meas - H @ xp
            S     = H @ Pp @ H.T + R
            K     = Pp @ H.T @ np.linalg.solve(S.T, np.eye(2)).T
            xu    = xp + K @ innov
            Pu    = (np.eye(len(xp)) - K @ H) @ Pp
            return xu, Pu, innov, S

        xu_cv, Pu_cv, innov_cv, S_cv = kf_update(xp_cv, Pp_cv, self.H_cv, self.R)
        xu_ca, Pu_ca, innov_ca, S_ca = kf_update(xp_ca, Pp_ca, self.H_ca, self.R)

        # ── Step 4: mode probability update ───────────────────
        L_cv = self._likelihood(innov_cv, S_cv)
        L_ca = self._likelihood(innov_ca, S_ca)
        L    = np.array([L_cv, L_ca])
        denom = float(c_j @ L)
        if denom < 1e-300 or not np.isfinite(denom):
            self.mu = np.array([0.5, 0.5])   # reset to uniform on numerical failure
        else:
            self.mu = c_j * L / denom
            s = self.mu.sum()
            self.mu = self.mu / s if s > 1e-300 else np.array([0.5, 0.5])

        # ── Step 5: store updated states ──────────────────────
        self.x_cv, self.P_cv = xu_cv, Pu_cv
        self.x_ca, self.P_ca = xu_ca, Pu_ca

        # ── Step 6: fused estimate ─────────────────────────────
        x_fused = self.mu[0] * xu_cv + self.mu[1] * xu_ca[:4]
        vx = x_fused[2]
        vz = x_fused[3]
        speed_ms  = np.hypot(vx, vz)
        return float(speed_ms * 3.6)   # km/h
