"""
CTRV (Constant Turn Rate and Velocity) Extended Kalman Filter — heading estimate.

Estimates a stable heading angle psi for a single track from its noisy
ground-plane (x, z) measurements. Unlike a raw atan2(dz, dx) position
difference — which swings wildly at low speed and under RT-DETR box jitter —
the EKF models the vehicle as moving along a circular arc with a constant
turn rate, so heading stays smooth in normal driving yet still snaps at impact.

State (5-vector):
    [x, z, v, psi, omega]
      x, z   : ground-plane position (metres); z is depth/forward along road
      v      : forward speed (m/s)
      psi    : heading (radians), psi=0 -> +z ("north"/forward),
               increasing toward +x.  vx = v*sin(psi), vz = v*cos(psi)
      omega  : yaw rate dpsi/dt (rad/s)

Nonlinear CTRV motion model (omega != 0):
    x'   = x + (v/omega)*( cos(psi)         - cos(psi+omega*dt) )
    z'   = z + (v/omega)*( sin(psi+omega*dt) - sin(psi)         )
    v'   = v
    psi' = psi + omega*dt
    omega' = omega
Straight-line limit (|omega| < EPS) integrates velocity directly.

Motion gate (heading is undefined for a near-stationary vehicle, and a parked
car's position jitter would otherwise spin the yaw-rate state and wind the
heading up by full rotations): a debounced state machine on the NET
displacement over a short window declares the vehicle "moving" only after N_ON
sustained frames above V_ON (rejecting isolated jitter spikes) and reverts
below V_OFF. While not moving, update() HOLDS the last good heading and zeroes
the yaw rate; while moving it trusts the EKF, seeding the heading from the
net-displacement direction on the first real motion. Yaw rate is clamped to
OMEGA_MAX so a single noise spike can never wind up.

Usage:
    f = CTRVFilter(dt=1/30)
    f.init(x0, z0)
    for each frame:
        heading_deg = f.update(x_m, z_m)   # heading in degrees [0, 360)
        rate_deg_s  = f.yaw_rate_deg       # signed yaw rate (deg/s)
        speed_kmh   = f.speed_kmh
"""

import numpy as np
from collections import deque

EPS            = 1e-4    # |omega| below this -> straight-line branch
V_ON           = 2.5     # m/s; sustained net-speed to declare "moving"
V_OFF          = 1.5     # m/s; net-speed below this -> back to "stationary"
N_ON           = 8       # consecutive frames above V_ON before switching on
N_WARMUP       = 10      # frames after motion onset to pin yaw=0 (kills overshoot)
OMEGA_MAX      = 1.2     # rad/s; physical cap on yaw rate (~69 deg/s)


class CTRVFilter:
    def __init__(self, dt: float = 1 / 30.0,
                 sigma_a: float = 3.0,
                 sigma_yaw: float = 0.3,
                 r: float = 0.5,
                 move_window: int = 7):
        """
        dt          : seconds between frames
        sigma_a     : longitudinal acceleration process-noise std (m/s^2)
        sigma_yaw   : yaw-acceleration process-noise std (rad/s^2)
        r           : measurement-noise std (metres)
        move_window : frames over which NET displacement decides moving vs
                      stationary (net cancels per-frame jitter)
        """
        self.dt        = dt
        self.sigma_a   = sigma_a
        self.sigma_yaw = sigma_yaw
        self.R         = r**2 * np.eye(2)
        self.H         = np.zeros((2, 5)); self.H[0, 0] = 1.0; self.H[1, 1] = 1.0
        self._pos      = deque(maxlen=move_window)   # (x, z) for the motion gate
        self._initialized = False

    # ── lifecycle ──────────────────────────────────────────────
    def init(self, x0: float, z0: float):
        # [x, z, v, psi, omega]
        self.x = np.array([x0, z0, 0.0, 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, 4.0, (np.pi)**2, 1.0])
        self._pos.clear()
        self._pos.append((x0, z0))
        self._last_heading = 0.0     # held while stationary
        self._has_heading  = False   # seeded once the vehicle first moves
        self._moving       = False   # debounced motion state
        self._on_count     = 0       # consecutive frames above V_ON
        self._warmup       = 0       # frames remaining to pin yaw=0 after onset
        self._initialized  = True

    # ── readouts ───────────────────────────────────────────────
    @property
    def has_heading(self) -> bool:
        """False during the onset delay before the first real heading is seeded
        (the held value is an arbitrary init, not a measured direction)."""
        return self._has_heading

    @property
    def heading_deg(self) -> float:
        return float(np.degrees(self.x[3]) % 360.0)

    @property
    def yaw_rate_deg(self) -> float:
        return float(np.degrees(self.x[4]))

    @property
    def speed_kmh(self) -> float:
        return float(self.x[2] * 3.6)

    # ── motion model + Jacobian ────────────────────────────────
    def _f_and_F(self):
        x, z, v, psi, omega = self.x
        dt = self.dt
        F  = np.eye(5)

        if abs(omega) < EPS:
            # straight-line limit
            s, c = np.sin(psi), np.cos(psi)
            xp = x + v * s * dt
            zp = z + v * c * dt
            xn = np.array([xp, zp, v, psi + omega * dt, omega])

            F[0, 2] = s * dt
            F[0, 3] = v * c * dt
            F[0, 4] = 0.5 * v * dt**2 * c     # 2nd-order omega sensitivity
            F[1, 2] = c * dt
            F[1, 3] = -v * s * dt
            F[1, 4] = -0.5 * v * dt**2 * s
            F[3, 4] = dt
        else:
            a = psi
            b = psi + omega * dt
            sa, ca = np.sin(a), np.cos(a)
            sb, cb = np.sin(b), np.cos(b)
            v_w = v / omega

            xp = x + v_w * (ca - cb)
            zp = z + v_w * (sb - sa)
            xn = np.array([xp, zp, v, b, omega])

            F[0, 2] = (ca - cb) / omega
            F[0, 3] = v_w * (-sa + sb)
            F[0, 4] = (-v / omega**2) * (ca - cb) + v_w * sb * dt
            F[1, 2] = (sb - sa) / omega
            F[1, 3] = v_w * (cb - ca)
            F[1, 4] = (-v / omega**2) * (sb - sa) + v_w * cb * dt
            F[3, 4] = dt

        return xn, F

    def _Q(self):
        """Process noise from longitudinal accel + yaw accel, mapped to state."""
        dt  = self.dt
        psi = self.x[3]
        s, c = np.sin(psi), np.cos(psi)
        # noise gain for linear acceleration nu_a
        Ga = np.array([0.5 * dt**2 * s, 0.5 * dt**2 * c, dt, 0.0, 0.0])
        # noise gain for yaw acceleration nu_psi
        Gp = np.array([0.0, 0.0, 0.0, 0.5 * dt**2, dt])
        return (np.outer(Ga, Ga) * self.sigma_a**2 +
                np.outer(Gp, Gp) * self.sigma_yaw**2)

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap to (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ── main step ──────────────────────────────────────────────
    def update(self, x_m: float, z_m: float) -> float:
        """Feed one measurement, return heading in degrees [0, 360)."""
        if not self._initialized:
            self.init(x_m, z_m)
            return 0.0

        # ── predict ────────────────────────────────────────────
        xp, F = self._f_and_F()
        xp[3] = self._wrap(xp[3])
        Pp = F @ self.P @ F.T + self._Q()

        # ── update (linear measurement on x, z) ────────────────
        z_meas = np.array([x_m, z_m])
        innov  = z_meas - self.H @ xp
        S      = self.H @ Pp @ self.H.T + self.R
        K      = Pp @ self.H.T @ np.linalg.solve(S.T, np.eye(2)).T
        xu     = xp + K @ innov
        xu[3]  = self._wrap(xu[3])
        Pu     = (np.eye(5) - K @ self.H) @ Pp
        # keep speed non-negative; a sign flip would alias into heading
        if xu[2] < 0:
            xu[2]  = -xu[2]
            xu[3]  = self._wrap(xu[3] + np.pi)
        # clamp yaw rate to a physical bound so a noise spike can't wind up
        xu[4]  = float(np.clip(xu[4], -OMEGA_MAX, OMEGA_MAX))
        self.x, self.P = xu, Pu

        self._pos.append((x_m, z_m))

        # ── debounced motion gate (net displacement cancels jitter) ──
        # A stationary car's jitter spikes net-speed above any instantaneous
        # threshold, so require N_ON sustained frames to switch ON (rejecting
        # isolated spikes) and switch OFF as soon as motion dies.
        x0, z0 = self._pos[0]
        x1, z1 = self._pos[-1]
        win_dt = (len(self._pos) - 1) * self.dt
        net_disp  = np.hypot(x1 - x0, z1 - z0)
        net_speed = net_disp / win_dt if win_dt > 1e-6 else 0.0

        if net_speed > V_ON:
            self._on_count += 1
            if self._on_count >= N_ON:
                self._moving = True
        elif net_speed < V_OFF:
            self._on_count = 0
            self._moving   = False

        if not self._moving:
            # Stationary / noise-dominated: heading is geometrically undefined.
            # Hold the last good heading and kill yaw so nothing winds up.
            self.x[3] = self._last_heading
            self.x[4] = 0.0
            return float(np.degrees(self._last_heading) % 360.0)

        # Moving: trust the EKF. On the first real motion, seed heading from the
        # net-displacement direction so it doesn't slew from a stale held value.
        # Also COLLAPSE the heading/yaw covariance so the EKF trusts the seed
        # instead of over-correcting toward the first noisy positions — that
        # over-correction is what produced the onset overshoot/swing.
        # Onset handling.  For the first N_WARMUP frames of motion, report the
        # GEOMETRIC net-displacement heading and pin the EKF to it (yaw=0). The
        # geometric heading can't overshoot — its baseline just grows more
        # accurate — whereas letting the EKF chase the first noisy positions
        # builds an omega that swings the heading past the true direction. After
        # warmup we hand off to the EKF, now seeded at the right heading.
        if not self._has_heading or self._warmup > 0:
            self.x[3] = np.arctan2(x1 - x0, z1 - z0)   # 0 -> +z, toward +x
            self.x[4] = 0.0
            self.P[3, 3] = np.radians(8.0) ** 2
            self.P[4, 4] = 0.05
            self.P[3, :3] = self.P[:3, 3] = 0.0
            self.P[3, 4]  = self.P[4, 3] = 0.0
            self.P[4, :3] = self.P[:3, 4] = 0.0
            if not self._has_heading:
                self._has_heading = True
                self._warmup = N_WARMUP
            else:
                self._warmup -= 1
        self._last_heading = self.x[3]
        return self.heading_deg
