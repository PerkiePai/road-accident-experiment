"""
CP5 — Per-track accident state machine.

Watches yaw rate (from CTRVFilter) and speed deceleration per canonical ID.

Three trigger conditions (any one fires):
  1. Sustained yaw: |yaw_deg_s| > YAW_SPIKE_DEG_S for N_CONFIRM consecutive frames
  2. Sudden stop:   decel > DECEL_ONLY_THRESH  (no yaw required — rear/frontal hits)
  3. Combo:         |yaw_deg_s| > YAW_SPIKE_DEG_S AND decel > DECEL_COMBO_THRESH

Deceleration is computed over a SHORT window (DECEL_WINDOW_S ~ 5 frames) to catch
the sharp velocity drop at impact, which a longer window averages away.

Auto-clear (ACCIDENT -> NORMAL):
  |yaw_deg_s| < YAW_CLEAR_DEG_S for CLEAR_FRAMES consecutive frames
  AND deceleration < DECEL_ONLY_THRESH / 2

No minimum speed floor — a stationary vehicle hit from behind also shows a yaw spike.
"""

from collections import deque

YAW_SPIKE_DEG_S     = 10.0   # |yaw| above this -> yaw-spike condition
YAW_CLEAR_DEG_S     =  5.0   # |yaw| below this (hysteresis) -> clearing
DECEL_ONLY_THRESH   = 50.0   # km/h/s: sudden-stop trigger (no yaw needed)
DECEL_COMBO_THRESH  = 20.0   # km/h/s: lower bar when combined with a yaw spike
DECEL_WINDOW_S      =  0.15  # seconds (~5 frames) — short to capture sharp transient
N_CONFIRM           =  3     # consecutive frames above YAW_SPIKE to confirm yaw trigger
CLEAR_FRAMES        = 20     # consecutive normal frames required to auto-clear


class AccidentDetector:
    """Independent accident state machine for every canonical track ID."""

    def __init__(self, fps: float):
        self._fps    = fps
        self._states = {}   # cid -> state dict

    def _get(self, cid: int) -> dict:
        if cid not in self._states:
            n = max(2, int(self._fps * DECEL_WINDOW_S))
            self._states[cid] = {
                'triggered':   False,
                'yaw_count':   0,
                'clear_count': 0,
                'speed_buf':   deque(maxlen=n),  # (t, speed_kmh)
                'trigger_t':   None,
                'trigger_spd': None,
                'trigger_yaw': None,
            }
        return self._states[cid]

    def update(self, cid: int, t: float, yaw_deg_s: float, speed_kmh: float):
        """Feed one frame for this track.

        Returns (is_accident, just_triggered, just_cleared).
        """
        st = self._get(cid)
        st['speed_buf'].append((t, speed_kmh))

        buf   = list(st['speed_buf'])
        dt    = buf[-1][0] - buf[0][0]
        decel = (-(buf[-1][1] - buf[0][1]) / dt) if dt > 1e-3 else 0.0  # +ve = decelerating

        yaw_abs        = abs(yaw_deg_s)
        just_triggered = False
        just_cleared   = False

        if not st['triggered']:
            if yaw_abs > YAW_SPIKE_DEG_S:
                st['yaw_count'] += 1
            else:
                st['yaw_count'] = 0

            # Any of three conditions triggers
            yaw_sustained = st['yaw_count'] >= N_CONFIRM
            sudden_stop   = decel > DECEL_ONLY_THRESH
            combo         = yaw_abs > YAW_SPIKE_DEG_S and decel > DECEL_COMBO_THRESH

            if yaw_sustained or sudden_stop or combo:
                st['triggered']   = True
                st['clear_count'] = 0
                st['trigger_t']   = t
                st['trigger_spd'] = speed_kmh
                st['trigger_yaw'] = yaw_deg_s
                just_triggered    = True
        else:
            # Clear: sustained low yaw AND decel below half the sudden-stop threshold
            if yaw_abs < YAW_CLEAR_DEG_S and decel < DECEL_ONLY_THRESH * 0.5:
                st['clear_count'] += 1
            else:
                st['clear_count'] = 0

            if st['clear_count'] >= CLEAR_FRAMES:
                st['triggered']   = False
                st['yaw_count']   = 0
                st['clear_count'] = 0
                just_cleared      = True

        return st['triggered'], just_triggered, just_cleared

    def is_accident(self, cid: int) -> bool:
        return self._states.get(cid, {}).get('triggered', False)

    def trigger_info(self, cid: int):
        """Return (trigger_t, trigger_spd, trigger_yaw) for the last trigger event."""
        st = self._states.get(cid, {})
        return st.get('trigger_t'), st.get('trigger_spd'), st.get('trigger_yaw')

    def remove(self, cid: int):
        self._states.pop(cid, None)
