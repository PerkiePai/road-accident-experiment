"""
CP5 — Per-track accident state machine.

Watches yaw rate (from CTRVFilter) and speed deceleration per canonical ID.

Three trigger conditions (any one fires):
  1. Sustained yaw: |yaw_deg_s| > YAW_SPIKE_DEG_S for N_CONFIRM consecutive frames
  2. Sudden stop:   DECEL_ONLY_THRESH < decel < DECEL_MAX_PHYS (no yaw — rear/frontal hits)
  3. Combo:         |yaw_deg_s| > YAW_SPIKE_DEG_S AND decel > DECEL_COMBO_THRESH

Deceleration is computed over a SHORT window (DECEL_WINDOW_S ~ 5 frames) to catch the
sharp velocity drop at impact, which a longer window averages away.

Two gates protect the decel channel from track-birth artefacts (the two robustness
fixes added after analysing the fr4 trigger cluster):

  * Physics gate (DECEL_MAX_PHYS) — decel must fall in a believable BAND, not just
    exceed a floor.  A road vehicle cannot brake harder than ~9 g; a window-decel
    above DECEL_MAX_PHYS is a measurement blow-up (a bad first speed estimate at
    track birth, typically far-field where one pixel maps to many metres) rather
    than a real deceleration, so it is rejected.

  * Young-track guard (MIN_TRACK_FRAMES) — the EMA speed (alpha=0.1) lags, so a track
    born with a high-ish first reading bleeds the EMA downward for ~10 frames
    regardless of what the vehicle does, and the buffer fills at exactly the frame
    that slope is steepest.  That manufactured deceleration is why so many false
    triggers land on the first full-buffer frame.  Until a track has been seen
    MIN_TRACK_FRAMES times its EMA is not trusted and decel-based triggers are
    suppressed.  The yaw channel is age-INDEPENDENT (it needs no absolute speed),
    so a stationary car spun on impact still fires immediately.

Auto-clear (ACCIDENT -> NORMAL):
  |yaw_deg_s| < YAW_CLEAR_DEG_S for CLEAR_FRAMES consecutive frames
  AND deceleration < DECEL_ONLY_THRESH / 2

No minimum speed floor — a stationary vehicle hit from behind also shows a yaw spike.
"""

from collections import deque

YAW_SPIKE_DEG_S      = 15.0   # |yaw| above this -> yaw-spike condition (15 avoids normal turns)
YAW_CLEAR_DEG_S      =  5.0   # |yaw| below this (hysteresis) -> clearing
DECEL_ONLY_THRESH    = 55.0   # km/h/s: sudden-stop trigger floor (raised from 50 to avoid
                              # borderline EMA-settling false positives)
DECEL_MAX_PHYS       = 130.0  # km/h/s: sudden-stop CEILING — above this is a measurement
                              # blow-up (~9 g is past any real vehicle), not a deceleration
DECEL_COMBO_THRESH   = 20.0   # km/h/s: lower bar when combined with a yaw spike
DECEL_WINDOW_S       =  0.15  # seconds (~5 frames) — short to capture sharp transient
N_CONFIRM            =  3     # consecutive frames above YAW_SPIKE to confirm yaw trigger
CLEAR_FRAMES         = 20     # consecutive normal frames required to auto-clear
MIN_TRACK_FRAMES     =  7     # frames a track must be seen before its EMA-decel is trusted


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
                'frames_seen': 0,
                'speed_buf':   deque(maxlen=n),  # (t, speed_kmh)
                'trigger_t':    None,
                'trigger_spd':  None,
                'trigger_yaw':  None,
            }
        return self._states[cid]

    def update(self, cid: int, t: float, yaw_deg_s: float, speed_kmh: float):
        """Feed one frame for this track.

        Returns (is_accident, just_triggered, just_cleared).
        """
        st = self._get(cid)
        st['frames_seen'] += 1
        st['speed_buf'].append((t, speed_kmh))
        buf      = list(st['speed_buf'])
        buf_full = len(buf) >= st['speed_buf'].maxlen
        dt       = buf[-1][0] - buf[0][0]
        # Only compute decel when the buffer is full — a partial buffer (1-2 entries)
        # computes over just 1-2 frames (0.03-0.07s), amplifying tiny EMA noise into
        # apparent 60+ km/h/s transients that mimic a crash.
        decel = (-(buf[-1][1] - buf[0][1]) / dt) if (buf_full and dt > 1e-3) else 0.0

        yaw_abs        = abs(yaw_deg_s)
        just_triggered = False
        just_cleared   = False

        # The EMA-decel channel is only trusted once the track's EMA has had time
        # to converge (MIN_TRACK_FRAMES) and only inside the physically-plausible
        # band [DECEL_ONLY_THRESH, DECEL_MAX_PHYS).  The yaw channel is independent
        # of both — a stationary car spun on impact must still fire on frame 1.
        decel_trusted = (st['frames_seen'] >= MIN_TRACK_FRAMES
                         and DECEL_ONLY_THRESH < decel < DECEL_MAX_PHYS)
        decel_combo   = (st['frames_seen'] >= MIN_TRACK_FRAMES
                         and DECEL_COMBO_THRESH < decel < DECEL_MAX_PHYS)

        if not st['triggered']:
            if yaw_abs > YAW_SPIKE_DEG_S:
                st['yaw_count'] += 1
            else:
                st['yaw_count'] = 0

            yaw_sustained = st['yaw_count'] >= N_CONFIRM
            sudden_stop   = decel_trusted
            combo         = yaw_abs > YAW_SPIKE_DEG_S and decel_combo

            if yaw_sustained or sudden_stop or combo:
                st['triggered']   = True
                st['clear_count'] = 0
                st['trigger_t']   = t
                st['trigger_spd'] = speed_kmh
                st['trigger_yaw'] = yaw_deg_s
                just_triggered    = True
        else:
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
