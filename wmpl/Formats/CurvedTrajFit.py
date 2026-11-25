""" Compact curved (Keplerian + J2 + NRLMSISE drag) trajectory fitter for satellite re-entries.

Takes a PyLIG (linear) trajectory pickle solved WITH Monte Carlo uncertainties and fits the state vector
(x, y, z, vx, vy, vz) at the first observation time plus a constant B*, minimising the angular residuals
between the observed lines of sight and the propagated arc. Constraints:
    - r0, v0 bounded by the linear solution's Monte Carlo 95% confidence intervals
    - B* bounded to (1 +/- bstar_frac) * TLE B*
    - a time offset correction per station, bounded to +/- toffset_bound around PyLIG's offset (PyLIG
      aligns stations using distances along the straight, non-gravity-dropped line, which biases the
      offsets on long arcs); PyLIG's timing reference station stays fixed at 0
    - optionally (--tle_constrain), the fitted orbit back-propagated to the TLE epoch must have a geodetic
      altitude within --tle_alt_tol_km of the TLE altitude at its epoch

Outputs (prefixed with "curved_") mirror PyLIG's trajectory-dependent outputs: trajectory pickle, report,
angular/spatial residual plots, lags, velocities, lengths, plus a linear vs curved angular RMSD table.

Usage:
    python -m wmpl.Formats.CurvedTrajFit PATH/TO/name_mc_trajectory.pickle --tle_file TLE.txt --norad_id 12345 \
        [--tle_constrain]
"""

from __future__ import print_function, division, absolute_import

import argparse
import copy
import os
import pickle
import re
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import spaceweather as sw
from scipy.integrate import solve_ivp

from wmpl.PythonNRLMSISE00.nrlmsise_00 import gtd7
from wmpl.PythonNRLMSISE00.nrlmsise_00_header import lstCalc, nrlmsise_flags, nrlmsise_input, nrlmsise_output
from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Trajectory.Trajectory import calcSpatialResidual, applyGravityDrop
from wmpl.Utils.GeoidHeightEGM96 import mslToWGS84Height
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.Plotting import savePlot
from wmpl.Utils.Math import findClosestPoints
from wmpl.Utils.TrajConversions import cartesian2Geo, eci2RaDec, jd2Date, raDec2AltAz


# Physical constants (same values as CurvedTraj.py)
MU = 3.986004418e14       # m^3/s^2
J2 = 1.08263e-3
R_E = 6.3781e6            # m
RHO0 = 2.461e-8           # SGP4 B* reference density, 0.15696615 kg/(m^2 ER) in kg/m^3; B* in 1/m
TLE_ER_M = 6378135.0      # Earth radius used for TLE B* units (1/ER -> 1/m)
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
DEFAULT_SW_FILE = "/Users/tammydo/Documents/phd-uwo/space_debris/spaceweather5Y.txt"

# Trajectory.savePlots figure keys that depend on the trajectory solution and are kept
KEEP_PLOTS = ["all_spatial_total_residuals_height", "all_angular_residuals", "lags_all", "velocities", "lengths"]

_DRAG_FLAGS = nrlmsise_flags()
_DRAG_FLAGS.switches[0] = 0
for _i in range(1, 24):
    _DRAG_FLAGS.switches[_i] = 1


def julian_date(dt_obj):
    a = (14 - dt_obj.month) // 12
    y = dt_obj.year + 4800 - a
    m = dt_obj.month + 12 * a - 3
    jdn = (
        dt_obj.day
        + ((153 * m + 2) // 5)
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        - 32045
    )
    frac_day = (
        (dt_obj.hour - 12) / 24
        + dt_obj.minute / 1440
        + dt_obj.second / 86400
        + dt_obj.microsecond / 8.64e10
    )
    return jdn + frac_day


def gmst_angle(dt_obj):
    jd = julian_date(dt_obj)
    T = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T**2
        - (T**3) / 38710000.0
    )
    return np.deg2rad(gmst % 360.0)


def eci_to_ecef(r_eci, dt_obj):
    theta = gmst_angle(dt_obj)
    rot = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return rot @ r_eci


def ecef_to_llh(r_ecef):
    x, y, z = r_ecef
    b = WGS84_A * np.sqrt(1 - WGS84_E2)
    ep = np.sqrt((WGS84_A**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + ep**2 * b * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt / 1000.0


def keplerian_propagator(
    r0_eci,
    v0_eci,
    t_array,
    b0=0.0,
    t0_epoch=None,
    f107=None,
    f107A=None,
    ap_daily=None,
):
    """
    Propagate an orbit from (r0, v0) at t=0 to all times in t_array.
    Uses simple 2-body + J2. Returns array of ECI positions shape (N, 3).

    For production, replace with sgp4/poliastro/orekit for full perturbations.
    """

    y0 = np.concatenate([r0_eci, v0_eci])
    t_span = (0.0, t_array[-1])

    sol = solve_ivp(
        eom,
        t_span,
        y0,
        t_eval=t_array,
        method="DOP853",
        args=(b0, t0_epoch, f107, f107A, ap_daily),
        rtol=1e-10,
        atol=1e-12,
    )

    return sol.y[:3, :].T  # shape (N, 3), ECI positions in metres



def computeAngularRMSD(
    observations,
    r0,
    v0,
    b0_fit=0.0,
    t0_epoch=None,
    f107=None,
    f107A=None,
    ap_daily=None,
):
    """
    Compute per-station angular RMSD in arcseconds.
    Frame-independent — no h/v decomposition needed.
    """
    t0_global = min(obs.time_data[0] for obs in observations)
    results = {}

    for obs in observations:
        if obs.ignore_station:
            continue

        t_rel = obs.time_data - t0_global
        traj_eci = keplerian_propagator(
            r0, v0, t_rel, b0_fit, t0_epoch, f107, f107A, ap_daily
        )

        angles = []
        for j, (pos, stat, meas, ign) in enumerate(
            zip(traj_eci, obs.stat_eci_los, obs.meas_eci_los, obs.ignore_list)
        ):
            if ign:
                continue
            model_ray = pos - stat
            norm = np.linalg.norm(model_ray)
            if norm < 1:
                continue
            model_ray /= norm
            cos_a = np.clip(np.dot(meas, model_ray), -1, 1)
            angles.append(np.degrees(np.arccos(cos_a)) * 3600)  # arcseconds

        if angles:
            results[str(obs.station_id)] = np.sqrt(np.mean(np.array(angles) ** 2))

    print(f"Curved — per-station angular RMSD (arcsec):")
    for sid, rmsd in sorted(results.items()):
        print(f"  {sid:12s}: {rmsd:.3f} arcsec")

    return results


def computeAngularRMSDLinear(observations, traj_los, label=""):
    """Per-station angular RMSD of PyLIG's straight line, evaluated as a
    function of time — the same metric used for the curved solution."""
    t_hat = -traj_los.radiant_eci_mini/np.linalg.norm(traj_los.radiant_eci_mini)
    P0 = traj_los.state_vect_mini

    # PyLIG's own length-vs-time model: length = slope*t + intercept
    slope, intercept = traj_los.velocity_fit

    # Gravity drop parameters, set up exactly as in Trajectory.calcECIEqAltAz: time measured from the first
    # observation, r0 = largest geocentric distance of the first-point CPAs on the radiant line
    gravity = getattr(traj_los, "gravity_correction", False)
    if gravity:
        t0_grav = min(obs.time_data[0] for obs in traj_los.observations)
        r0_grav = max(
            np.linalg.norm(findClosestPoints(obs.stat_eci_los[0], obs.meas_eci_los[0], P0,
                                             traj_los.radiant_eci_mini)[1])
            for obs in traj_los.observations)
        v0z = traj_los.v0z if traj_los.v0z is not None else 0.0

    results = {}
    for obs in observations:
        if obs.ignore_station:
            continue
        m = obs.ignore_list == 0
        if not np.any(m):
            continue

        t = obs.time_data[m]
        length = slope*t + intercept
        a1, a2 = traj_los.jacchia_fit
        if a1 or a2:
            length = length - abs(a1)*np.exp(abs(a2)*t)
        pos = P0[None, :] + t_hat[None, :]*length[:, None]
        if gravity:
            pos = np.array([applyGravityDrop(p, tk - t0_grav, r0_grav, traj_los.gravity_factor, v0z)
                            for p, tk in zip(pos, t)])

        rays = pos - obs.stat_eci_los[m]
        rays = rays/np.linalg.norm(rays, axis=1)[:, None]
        cos_a = np.clip(np.einsum("ij,ij->i", obs.meas_eci_los[m], rays), -1, 1)
        results[str(obs.station_id)] = float(
            np.sqrt(np.mean((np.degrees(np.arccos(cos_a))*3600.0)**2)))

    print(f"\n{label} — per-station angular RMSD (arcsec):")
    for sid, rmsd in sorted(results.items()):
        print(f"  {sid:12s}: {rmsd:.3f} arcsec")

    return results


# Reused from wmpl/Formats/CurvedTraj.py:eom — unchanged physics (2-body + J2 + NRLMSISE-00 drag with a
# constant B*). Must stay identical to CurvedTraj.eom because computeAngularRMSD (imported) uses that one.
def eom(t, y, b0, t0_epoch, f107, f107A, ap_daily):
    r, v = y[:3], y[3:]
    r_mag = np.sqrt(r @ r)

    a = -MU/r_mag**3*r

    z2, r2 = r[2]**2, r_mag**2
    j2_fac = 1.5*J2*MU*R_E**2/r_mag**5
    a_j2 = j2_fac*np.array([r[0]*(5*z2/r2 - 1), r[1]*(5*z2/r2 - 1), r[2]*(5*z2/r2 - 3)])

    a_drag = np.zeros(3)
    vmag2 = v @ v
    if b0 and vmag2 > 0.0:
        epoch = t0_epoch + timedelta(seconds=float(t))
        lat, lon, alt = ecef_to_llh(eci_to_ecef(r, epoch))

        inp = nrlmsise_input()
        inp.f107A, inp.f107, inp.ap = f107A, f107, ap_daily
        inp.doy = epoch.timetuple().tm_yday
        inp.sec = epoch.hour*3600 + epoch.minute*60 + epoch.second + epoch.microsecond/1e6
        inp.alt, inp.g_lat, inp.g_long = alt, lat, lon
        lstCalc(inp)
        out = nrlmsise_output()
        gtd7(inp, _DRAG_FLAGS, out)
        rho = out.d[5]*1000.0

        a_drag = -(rho/RHO0)*abs(b0)*np.sqrt(vmag2)*v

    return np.concatenate([v, a + a_j2 + a_drag])


# Reused from wmpl/Formats/CurvedTraj.py:_build_dense_trajectory — dropped the arc-length grid, returns
# only a dense interpolant (6 x N state); integrates both backwards (to t_lo <= 0) and forwards (to
# t_hi >= 0) from the state epoch, needed because time offsets can move points before the epoch
def propagateDense(r0, v0, t_lo, t_hi, b0, t0_epoch, sw_vals, rtol=1e-9):
    sols = []
    for t_end in (min(t_lo, -1e-3), max(t_hi, 1e-3)):
        sol = solve_ivp(eom, (0.0, t_end), np.concatenate([r0, v0]), method="DOP853",
                        args=(b0, t0_epoch, *sw_vals), dense_output=True,
                        rtol=rtol, atol=[1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
        if not sol.success:
            return None
        sols.append(sol.sol)

    def dense(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty((6, len(t)))
        back = t < 0
        if back.any():
            out[:, back] = sols[0](t[back])
        if (~back).any():
            out[:, ~back] = sols[1](t[~back])
        return out

    return dense


def findReferenceStation(traj):
    """ Index of PyLIG's timing reference station (fixed offset of 0), as chosen in
    Trajectory.estimateTimingAndVelocity: the station with the most measurements. """
    for i, obs in enumerate(traj.observations):
        val = traj.stations_time_dict.get(str(obs.station_id), True)
        if not isinstance(val, bool) and val == 0:
            return i
    return int(np.argmax([obs.kmeas for obs in traj.observations]))


def stateBoundsFromCI(traj, ci_scale=1):
    """ Bounds on (x, y, z, vx, vy, vz) as offsets from PyLIG's reported (lowest timing residual) solution,
    taken from the Monte Carlo confidence intervals (traj.uncertainties.*_ci, 95% by default). Both sides
    of each interval are multiplied by ci_scale about the reported solution.

    PyLIG's vx/vy/vz (and their CIs) are components of v_init*radiant_eci_mini, i.e. they point back to
    the radiant; the velocity of motion is the negative, so those intervals are flipped. The CI is the
    percentile range of all MC solutions and is not centred on the reported solution, so it can exclude
    it; such an interval is extended past the reported solution by 10% of its width. The fit starts at the
    reported solution and must start strictly inside the bounds: least_squares (trf) stalls with
    vanishing steps when started exactly on a bound (seen with SciPy 1.15).

    Return:
        (lo, hi, widened): offsets from the reported solution (m, m/s) and names of widened intervals.
    """
    un = traj.uncertainties
    v_rad = traj.v_init*traj.radiant_eci_mini  # PyLIG's reported "velocity" (towards the radiant)
    lo, hi = np.zeros(6), np.zeros(6)
    for k, name in enumerate(["x", "y", "z"]):
        a, b = getattr(un, name + "_ci")
        lo[k], hi[k] = a - traj.state_vect_mini[k], b - traj.state_vect_mini[k]
    for k, name in enumerate(["vx", "vy", "vz"]):
        a, b = getattr(un, name + "_ci")
        lo[3 + k], hi[3 + k] = v_rad[k] - b, v_rad[k] - a

    names = ["x", "y", "z", "vx", "vy", "vz"]
    widened = [n for n, l, h in zip(names, lo, hi) if l >= 0 or h <= 0]
    margin = 0.1*(hi - lo)
    lo = np.where(lo >= 0, -margin, lo)
    hi = np.where(hi <= 0, margin, hi)
    return ci_scale*lo, ci_scale*hi, widened


def cameraID(station_id):
    """ Physical camera of an observation: PyLIG suffixes repeat arcs from one camera as NZ000D_2, NZ000D_3... """
    return re.sub(r"_\d+$", "", str(station_id))


# Reused from wmpl/Formats/CurvedTraj.py:get_space_weather — space weather file path is now an argument
def getSpaceWeather(epoch_dt, sw_file):
    """ Return (f107 previous day, f107 81-day centred average, daily Ap) for NRLMSISE-00. """
    df = sw.sw_daily(swpath_5y=sw_file, update=False)
    row_today = df.loc[epoch_dt.strftime("%Y-%m-%d")]
    row_prev = df.loc[(epoch_dt - timedelta(days=1)).strftime("%Y-%m-%d")]
    return float(row_prev["f107_obs"]), float(row_today["f107_81ctr_obs"]), float(row_today["Apavg"])


# Reused from wmpl/Formats/CurvedTraj.py:angleResidualsOrbitalArc — dropped sigma normalisation and
# max_fit_seconds, skips ignored stations, applies per-station time offsets dts (s, one per observation),
# returns per-point angular residuals in arcsec
def angularResiduals(r0, v0, b0, observations, dts, t0_global, t0_epoch, sw_vals):
    t_lo = min(obs.time_data[0] + dt for obs, dt in zip(observations, dts)) - t0_global
    t_hi = max(obs.time_data[-1] + dt for obs, dt in zip(observations, dts)) - t0_global
    dense = propagateDense(r0, v0, t_lo, t_hi, b0, t0_epoch, sw_vals)
    n_pts = sum(int(np.sum(obs.ignore_list == 0)) for obs in observations if not obs.ignore_station)
    if dense is None:
        return np.full(n_pts, 1e6)

    res = []
    for obs, dt in zip(observations, dts):
        if obs.ignore_station:
            continue
        m = obs.ignore_list == 0
        pos = dense(obs.time_data[m] + dt - t0_global)[:3].T
        rays = pos - obs.stat_eci_los[m]
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        cos_a = np.clip(np.einsum("ij,ij->i", obs.meas_eci_los[m], rays), -1, 1)
        res.append(np.degrees(np.arccos(cos_a))*3600.0)

    return np.concatenate(res)


# Reused from phd-uwo utils/data_analysis_utils.py:load_tle_satellite + get_tle_bstar — returns the
# epoch JD and the geodetic altitude at epoch
def loadTLE(tle_path, norad_id, event_jd):
    from skyfield.api import load, wgs84
    from skyfield.iokit import parse_tle_file

    ts = load.timescale()
    with load.open(str(tle_path)) as tle_file:
        satellites = list(parse_tle_file(tle_file, ts))
        sat = {sat.model.satnum: sat for sat in satellites}[norad_id]

    bstar_si = abs(sat.model.bstar)/TLE_ER_M
    alt_km = wgs84.height_of(sat.at(sat.epoch)).km

    return sat, bstar_si, sat.epoch.tt, alt_km


def tleAltitudeDiff(r0, v0, b0, t0_jd, t0_epoch, sw_vals, tle_jd, tle_alt_km):
    """ Back-propagate (r0, v0) to the TLE epoch and return (fit altitude - TLE altitude) in km. """
    dt = (tle_jd - t0_jd)*86400.0
    dense = propagateDense(r0, v0, min(dt, 0.0), max(dt, 0.0), b0, t0_epoch, sw_vals, rtol=1e-8)
    if dense is None:
        return np.inf
    _, _, ht = cartesian2Geo(tle_jd, *dense(dt)[:3, 0])
    return ht/1000.0 - tle_alt_km


def fitCurved(traj, bstar0, t0_global, t0_jd, t0_epoch, sw_vals, bstar_frac, toffset_bound=0.5,
              tle=None, tle_alt_tol_km=10.0, ci_scale=1):
    """ Bounded least-squares fit of (r0, v0, B*) and per-station time offset corrections to the angular
    residuals.

    ci_scale: integer multiplier on the linear MC confidence intervals used as state bounds (1 = as is).

    toffset_bound: max change (s) to each camera's time offset. One offset per physical camera (all of its
        arcs share one clock), starting from the point-weighted mean of PyLIG's offsets for its arcs; the
        camera of PyLIG's reference station stays fixed at the reference offset. 0 keeps PyLIG's offsets.
    tle: None, or (tle_jd, tle_alt_km) to enforce |dh| <= tle_alt_tol_km at the TLE epoch.
    """
    # Seed: position on PyLIG's line at t0_global using its own length-vs-time model (same as
    # computeAngularRMSDLinear) - state_vect_mini itself is not the position at t0_global
    t_hat = -traj.radiant_eci_mini/np.linalg.norm(traj.radiant_eci_mini)
    slope, intercept = traj.velocity_fit
    length0 = slope*t0_global + intercept
    a1, a2 = traj.jacchia_fit
    if a1 or a2:
        length0 -= abs(a1)*np.exp(abs(a2)*t0_global)
    x0 = np.concatenate([traj.state_vect_mini + t_hat*length0, t_hat*slope, [bstar0]])

    obs = traj.observations

    # One time offset per physical camera; the camera of PyLIG's timing reference stays fixed
    ref = findReferenceStation(traj)
    cams = np.array([cameraID(o.station_id) for o in obs])
    pylig = np.asarray(traj.time_diffs_final, dtype=float)
    used = [i for i, o in enumerate(obs) if not o.ignore_station]
    free_cams = []
    base = np.zeros(len(obs))  # correction that puts all arcs of a camera on the camera's starting offset
    if toffset_bound > 0:
        free_cams = list(dict.fromkeys(cams[i] for i in used if cams[i] != cams[ref]))
        for c in set(cams):
            idx = [i for i in range(len(obs)) if cams[i] == c]
            if c == cams[ref]:
                start = pylig[ref]
            else:
                w = [max(int(np.sum(obs[i].ignore_list == 0)), 1) for i in idx]
                start = np.average(pylig[idx], weights=w)
            base[idx] = start - pylig[idx]
    dt_scale = 0.05  # s per normalised unit

    # State bounds from PyLIG's MC 95% confidence intervals (times ci_scale); each component is normalised
    # by its half-width
    ci_lo, ci_hi, ci_widened = stateBoundsFromCI(traj, ci_scale)
    if ci_widened:
        print("Warning: MC confidence interval excludes PyLIG's reported value for {}; extended past "
              "it by 10% of its width".format(", ".join(ci_widened)))
    state_scale = np.maximum((ci_hi - ci_lo)/2.0, [1.0, 1.0, 1.0, 0.01, 0.01, 0.01])

    n_dt = len(free_cams)
    scale = np.r_[state_scale, bstar0, np.full(n_dt, dt_scale)].astype(float)
    lo = np.r_[ci_lo/state_scale, -bstar_frac, np.full(n_dt, -toffset_bound/dt_scale)]
    hi = np.r_[ci_hi/state_scale, bstar_frac, np.full(n_dt, toffset_bound/dt_scale)]
    n_par = len(scale)
    x0 = np.r_[x0, np.zeros(n_dt)]

    expand = lambda p: x0 + p*scale

    def offsets(x):
        dts = base.copy()
        for k, c in enumerate(free_cams):
            dts[cams == c] += x[7 + k]
        return dts

    def resid(p):
        x = expand(p)
        return angularResiduals(x[:3], x[3:6], x[6], obs, offsets(x), t0_global, t0_epoch, sw_vals)

    print("Timing reference camera (fixed): {}; fitting one offset for each of {} cameras (+/-{} s): {}".format(
        cams[ref], n_dt, toffset_bound, ", ".join(free_cams)))

    seed_res = resid(np.zeros(n_par))
    print("Seed (linear state propagated on the curved model): angular RMS = {:.2f} arcsec".format(
        np.sqrt(np.mean(seed_res**2))))

    lsq_kwargs = dict(bounds=(lo, hi), method="trf", x_scale=1.0, diff_step=1e-2, xtol=1e-10, ftol=1e-10,
                      gtol=1e-10, max_nfev=500, verbose=1)

    print("Stage 1: angular residual fit")
    res = scipy.optimize.least_squares(resid, np.zeros(n_par), **lsq_kwargs)
    x = expand(res.x)
    info = {"stage": 1, "status": res.status, "message": res.message, "cost": res.cost, "nfev": res.nfev}

    dh = None
    if tle is not None:
        tle_jd, tle_alt_km = tle
        dh = tleAltitudeDiff(x[:3], x[3:6], x[6], t0_jd, t0_epoch, sw_vals, tle_jd, tle_alt_km)
        print("Stage 1 altitude at TLE epoch differs from TLE by {:.2f} km".format(dh))

        if not abs(dh) <= tle_alt_tol_km:
            # Hinge penalty, weighted so 1 km of violation outweighs 10x the whole stage-1 angular cost
            w = np.sqrt(20.0*res.cost)

            def resid_tle(p):
                xp = expand(p)
                dhp = tleAltitudeDiff(xp[:3], xp[3:6], xp[6], t0_jd, t0_epoch, sw_vals, tle_jd, tle_alt_km)
                excess = abs(dhp) - tle_alt_tol_km if np.isfinite(dhp) else 1e3
                return np.r_[resid(p), w*max(0.0, excess)]

            print("Stage 2: refit with TLE altitude constraint")
            res = scipy.optimize.least_squares(resid_tle, res.x, **lsq_kwargs)
            x = expand(res.x)
            dh = tleAltitudeDiff(x[:3], x[3:6], x[6], t0_jd, t0_epoch, sw_vals, tle_jd, tle_alt_km)
            info = {"stage": 2, "status": res.status, "message": res.message, "cost": res.cost,
                    "nfev": res.nfev}
            print("Stage 2 altitude at TLE epoch differs from TLE by {:.2f} km".format(dh))

    names = ["x", "y", "z", "vx", "vy", "vz", "B*"] + ["dt_" + c for c in free_cams]
    info["at_bound"] = [n for n, p, l, h in zip(names, res.x, lo, hi) if min(p - l, h - p) <= 1e-3*(h - l)]
    info["ci_widened"] = ci_widened
    info["dh_tle_km"] = dh
    info["x0"] = x0[:7]
    info["bounds_lo"], info["bounds_hi"] = expand(lo)[:7], expand(hi)[:7]
    info["toffset_ref_station"] = str(cams[ref])
    info["toffset_bound"] = toffset_bound

    return x[:3], x[3:6], x[6], offsets(x), info


# Reused from wmpl/Formats/CurvedTraj.py:computeCurvedTrajectoryMetrics + steps 1-4 of saveCurvedTrajectory
# — patches a copy of the Trajectory in place (instead of returning dicts), dropped the MergedObs block,
# uses Trajectory.calcSpatialResidual for h/v residuals, keeps PyLIG's radian conventions for model_* angles,
# and follows Trajectory.calcLLA for the per-station and trajectory begin/end/lowest points
def computeCurvedObservations(traj, r0, v0, b0, dts, t0_global, t0_epoch, sw_vals):
    traj_c = copy.deepcopy(traj)
    obs_all = traj_c.observations

    # Apply the fitted time offset corrections on top of PyLIG's (already applied) offsets
    for obs, dt in zip(obs_all, dts):
        obs.time_data = obs.time_data + dt
        obs.JD_data = obs.JD_data + dt/86400.0
    traj_c.time_diffs_final = np.asarray(traj.time_diffs_final, dtype=float) + dts

    t_lo = min(obs.time_data[0] for obs in obs_all) - t0_global
    t_hi = max(obs.time_data[-1] for obs in obs_all) - t0_global
    dense = propagateDense(r0, v0, t_lo, t_hi, b0, t0_epoch, sw_vals)

    # Arc length along the curved path on a fine grid, measured from r0 (t = 0)
    t_fine = np.linspace(min(t_lo, 0.0), t_hi, 5000)
    pos_fine = dense(t_fine)[:3].T
    arc_fine = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos_fine, axis=0), axis=1))]
    arc_fine -= np.interp(0.0, t_fine, arc_fine)

    for obs in obs_all:
        t_rel = obs.time_data - t0_global
        n = len(t_rel)

        # Model point at the observation time, used only for the (time-based) angular residuals = fit metric
        pos_t = dense(t_rel)[:3].T

        # Geometric model point, as in PyLIG: the point on the arc closest to each observed line of sight.
        # Used for spatial residuals, lat/lon/ht, lengths, lags and velocities so they reflect the data
        t_cpa = np.empty(n)
        for i, (stat, meas) in enumerate(zip(obs.stat_eci_los, obs.meas_eci_los)):
            d = pos_fine - stat
            d2 = np.einsum("ij,ij->i", d, d) - (d @ meas)**2
            k = int(np.clip(np.argmin(d2), 1, len(t_fine) - 2))
            # Parabolic refinement of the minimum between grid points
            den = d2[k - 1] - 2*d2[k] + d2[k + 1]
            shift = 0.5*(d2[k - 1] - d2[k + 1])/den if den > 0 else 0.0
            t_cpa[i] = t_fine[k] + np.clip(shift, -1, 1)*(t_fine[1] - t_fine[0])
        state = dense(t_cpa)
        pos, vel = state[:3].T, state[3:].T

        obs.model_eci = pos
        obs.state_vect_dist = np.interp(t_cpa, t_fine, arc_fine)
        obs.length = obs.state_vect_dist - obs.state_vect_dist[0]

        for name in ["model_ra", "model_dec", "model_azim", "model_elev", "model_lat", "model_lon", "model_ht",
                     "model_range", "meas_lat", "meas_lon", "meas_ht", "meas_range", "h_residuals",
                     "v_residuals", "ang_res"]:
            setattr(obs, name, np.zeros(n))

        for i, (jd, p, v, stat, meas) in enumerate(zip(obs.JD_data, pos, vel, obs.stat_eci_los,
                                                       obs.meas_eci_los)):
            ray = p - stat
            obs.model_range[i] = np.linalg.norm(ray)
            obs.model_ra[i], obs.model_dec[i] = eci2RaDec(ray)
            obs.model_azim[i], obs.model_elev[i] = raDec2AltAz(obs.model_ra[i], obs.model_dec[i], jd, obs.lat,
                                                               obs.lon)
            obs.model_lat[i], obs.model_lon[i], obs.model_ht[i] = cartesian2Geo(jd, *p)

            # Observed CPA: point on the measured line of sight closest to the model point
            obs_cpa = stat + np.dot(ray, meas)*meas
            obs.meas_range[i] = np.linalg.norm(obs_cpa - stat)
            obs.meas_lat[i], obs.meas_lon[i], obs.meas_ht[i] = cartesian2Geo(jd, *obs_cpa)

            # PyLIG h/v residuals relative to the local tangent of the curved arc
            obs.h_residuals[i], obs.v_residuals[i] = calcSpatialResidual(traj.jdt_ref, jd, p,
                                                                         -v/np.linalg.norm(v), stat, meas)

            ray_t = pos_t[i] - stat
            obs.ang_res[i] = np.arccos(np.clip(np.dot(meas, ray_t/np.linalg.norm(ray_t)), -1, 1))

        if traj_c.meastype == 1:
            obs.model_fit1, obs.model_fit2 = obs.model_ra, obs.model_dec
        else:
            obs.model_fit1, obs.model_fit2 = obs.model_azim, obs.model_elev

        m = obs.ignore_list == 0 if not obs.ignore_station else np.ones(n, dtype=bool)
        obs.h_res_rms = np.sqrt(np.mean(obs.h_residuals[m]**2))
        obs.v_res_rms = np.sqrt(np.mean(obs.v_residuals[m]**2))
        obs.ang_res_std = np.sqrt(np.mean(obs.ang_res[m]**2))

        # Point-to-point velocities (same formula as Trajectory.calcVelocity)
        dt = np.diff(np.r_[0.0, obs.time_data])
        dt[dt == 0] = np.finfo(np.float64).eps
        obs.velocities = np.diff(np.r_[0.0, obs.state_vect_dist])/dt
        obs.velocities_prev_point = obs.velocities.copy()

        # Begin/end/lowest points (Trajectory.calcLLA)
        idx = np.where(m)[0]
        i_min = idx[np.argmin(obs.model_ht[idx])]
        for pre, k in [("rbeg", idx[0]), ("rend", idx[-1]), ("htmin", i_min)]:
            setattr(obs, pre + "_lat", obs.model_lat[k])
            setattr(obs, pre + "_lon", obs.model_lon[k])
            setattr(obs, pre + "_ele", obs.model_ht[k])
            setattr(obs, pre + "_jd", obs.JD_data[k])

    # Trajectory-level begin/end/lowest points (Trajectory.calcLLA)
    used = [obs for obs in obs_all if not obs.ignore_station]
    first = min(used, key=lambda o: np.min(o.state_vect_dist))
    last = max(used, key=lambda o: np.max(o.state_vect_dist))
    lowest = min(used, key=lambda o: o.htmin_ele)
    for pre, o in [("rbeg", first), ("rend", last), ("htmin", lowest)]:
        for suf in ["_lat", "_lon", "_ele", "_jd"]:
            setattr(traj_c, pre + suf, getattr(o, pre + suf))
        setattr(traj_c, pre + "_ele_wgs84", mslToWGS84Height(getattr(o, pre + "_lat"), getattr(o, pre + "_lon"),
                                                             getattr(o, pre + "_ele")))

    # Global velocity fit on the first v_init_part of points (CurvedTraj.saveCurvedTrajectory step 2) and lag
    t_all = np.concatenate([o.time_data[o.ignore_list == 0] for o in used])
    d_all = np.concatenate([o.state_vect_dist[o.ignore_list == 0] for o in used])
    srt = np.argsort(t_all)
    n_fit = max(4, int(traj_c.v_init_part*len(t_all)))
    traj_c.velocity_fit = np.polyfit(t_all[srt][:n_fit], d_all[srt][:n_fit], 1)
    for obs in obs_all:
        obs.lag_line = list(traj_c.velocity_fit)
        obs.lag = obs.state_vect_dist - np.polyval(traj_c.velocity_fit, obs.time_data)
        obs.jacchia_fit = np.zeros(2)

    # Trajectory-level state
    v0_norm = np.linalg.norm(v0)
    traj_c.state_vect_mini = r0
    traj_c.radiant_eci_mini = -v0/v0_norm
    traj_c.v_init = v0_norm
    traj_c.v_init_stddev = None
    traj_c.jacchia_fit = np.zeros(2)
    traj_c.show_jacchia = False
    traj_c.v_avg, traj_c.state_vect_avg, traj_c.jd_avg = traj_c.calcAverages(obs_all)
    traj_c.uncertainties = traj_c.uncertanties = None
    traj_c.state_vect_cov = traj_c.orbit_cov = None

    try:
        traj_c.orbit = calcOrbit(traj_c.radiant_eci_mini, traj_c.v_init, traj_c.v_avg, r0,
                                 traj.jdt_ref + t0_global/86400.0, stations_fixed=False, reference_init=True)
    except Exception as e:
        print("Warning: could not recompute the orbit for the curved state ({})".format(e))

    return traj_c


def saveCurvedPlots(traj_c, dir_path, prefix, plot_file_type):
    """ Save only the trajectory-dependent PyLIG plots, using Trajectory.savePlots(ret_figs=True). """
    figs = traj_c.savePlots(None, None, show_plots=False, ret_figs=True)
    for key, fig_bytes in figs.items():
        if key.startswith("spatial_residuals_"):
            name = "{}_{}_spatial_residuals".format(prefix, key[len("spatial_residuals_"):])
        elif key in KEEP_PLOTS:
            name = "{}_{}".format(prefix, key)
        else:
            continue
        fig = pickle.loads(fig_bytes)
        savePlot(fig, name + "." + plot_file_type, dir_path)
        plt.close(fig)


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Fit a curved (Keplerian + drag) arc to a PyLIG MC trajectory.")
    ap.add_argument("pickle_path", help="PyLIG *_trajectory.pickle solved with Monte Carlo uncertainties.")
    ap.add_argument("--tle_file", help="TLE file containing the object's TLE(s).")
    ap.add_argument("--norad_id", type=int, help="NORAD catalogue number of the object.")
    ap.add_argument("--bstar", type=float, help="B* initial guess in TLE units (1/ER); overrides the TLE B*.")
    ap.add_argument("--tle_constrain", action="store_true",
                    help="Require the orbit altitude at the TLE epoch to match the TLE altitude.")
    ap.add_argument("--tle_alt_tol_km", type=float, default=10.0, help="TLE altitude tolerance (km).")
    ap.add_argument("--ci_scale", type=int, default=1,
                    help="Integer multiplier on the linear MC 95%% confidence intervals used as the x, y, z, "
                    "vx, vy, vz bounds (scaled about PyLIG's solution). 1 by default.")
    ap.add_argument("--bstar_frac", type=float, default=0.5, help="B* bounds = (1 +/- bstar_frac) * B*_0.")
    ap.add_argument("--toffset_bound", type=float, default=0.5,
                    help="Max change (s) to each station's PyLIG time offset; 0 keeps PyLIG's offsets fixed.")
    ap.add_argument("--sw_file", default=DEFAULT_SW_FILE, help="CelesTrak 5-year space weather file.")
    ap.add_argument("--imgformat", default="png", help="Plot file format.")
    args = ap.parse_args()

    # --- Load and validate ---
    traj = loadPickle(*os.path.split(os.path.abspath(args.pickle_path)))
    if getattr(traj, "uncertainties", None) is None:
        ap.error("The pickle has no Monte Carlo uncertainties. Re-run PyLIG without --disablemc and use "
                 "'Monte Carlo/<name>_mc_trajectory.pickle'.")
    if args.ci_scale < 1:
        ap.error("--ci_scale must be an integer >= 1.")
    if args.tle_constrain and not (args.tle_file and args.norad_id):
        ap.error("--tle_constrain needs --tle_file and --norad_id.")
    if args.bstar is None and not (args.tle_file and args.norad_id):
        ap.error("Give --tle_file and --norad_id (B* from the TLE), or --bstar.")

    t0_global = min(obs.time_data[0] for obs in traj.observations)
    t0_jd = traj.jdt_ref + t0_global/86400.0
    t0_epoch = jd2Date(traj.jdt_ref, dt_obj=True) + timedelta(seconds=float(t0_global))
    sw_vals = getSpaceWeather(t0_epoch, args.sw_file)

    tle_info = None
    bstar_tle = None
    if args.tle_file:
        sat, bstar_tle, tle_jd, tle_alt_km = loadTLE(args.tle_file, args.norad_id, t0_jd)
        tle_info = (tle_jd, tle_alt_km)
        print("TLE epoch {} ({:.3f} h before the event), altitude {:.2f} km, B* = {:.4e} 1/m".format(
            jd2Date(tle_jd, dt_obj=True), (t0_jd - tle_jd)*24, tle_alt_km, bstar_tle))
    bstar0 = abs(args.bstar)/TLE_ER_M if args.bstar is not None else bstar_tle

    # --- Fit ---
    r0, v0, b0, dts, info = fitCurved(traj, bstar0, t0_global, t0_jd, t0_epoch, sw_vals,
                                      args.bstar_frac, toffset_bound=args.toffset_bound,
                                      tle=(tle_info if args.tle_constrain else None),
                                      tle_alt_tol_km=args.tle_alt_tol_km, ci_scale=args.ci_scale)
    if info["dh_tle_km"] is None and tle_info is not None:
        info["dh_tle_km"] = tleAltitudeDiff(r0, v0, b0, t0_jd, t0_epoch, sw_vals, *tle_info)
    print("B* = {:.4e} 1/m ({:.4e} 1/ER)".format(b0, b0*TLE_ER_M))

    # --- Curved trajectory (observations carry the fitted time offsets) ---
    traj_c = computeCurvedObservations(traj, r0, v0, b0, dts, t0_global, t0_epoch, sw_vals)

    # --- Angular RMSD comparison ---
    angular_linear = computeAngularRMSDLinear(traj.observations, traj, "Linear (PyLIG) - " + traj.file_name)

    # computeAngularRMSD puts the state at the earliest (now time-shifted) observation, so hand it the fitted
    # state propagated to that time
    dt_first = min(obs.time_data[0] for obs in traj_c.observations) - t0_global
    state_first = propagateDense(r0, v0, min(dt_first, 0.0), max(dt_first, 0.0), b0, t0_epoch,
                                 sw_vals)(dt_first)[:, 0]
    angular_curved = computeAngularRMSD(traj_c.observations, state_first[:3], state_first[3:], b0,
                                        t0_epoch + timedelta(seconds=float(dt_first)), *sw_vals)

    dir_path = traj.output_dir
    prefix = "curved_" + traj.file_name
    state_epoch_dt = jd2Date(t0_jd, dt_obj=True)

    # Reused from wmpl/Formats/CurvedTraj.py __main__ "# --- Build the side-by-side comparison table" —
    # added B*, TLE altitude check and at-bound lines
    comparison_lines = [
        "Linear vs. Curved Trajectory Angular RMSD Comparison",
        "=" * 60,
        "",
        "State epoch : {:s} UTC".format(state_epoch_dt.strftime("%Y-%m-%d %H:%M:%S.%f")),
        "Position (m): {:s}".format(str(r0)),
        "Velocity m/s: {:s}".format(str(v0)),
        "B* (1/m)    : {:.4e}   (initial {:.4e}, bounds +/-{:.0f}%)".format(b0, bstar0, 100*args.bstar_frac),
        "TLE dh (km) : {}".format("n/a" if info["dh_tle_km"] is None else "{:.2f}{}".format(
            info["dh_tle_km"], " (constrained)" if args.tle_constrain else "")),
        "At bound    : {}".format(", ".join(info["at_bound"]) if info["at_bound"] else "none"),
        "",
        "{:<14s}  {:>16s}  {:>16s}  {}".format("Station", "Linear (arcsec)", "Curved (arcsec)", "Better?"),
    ]
    for sid in sorted(angular_linear):
        lin = angular_linear.get(sid, float("nan"))
        cur = angular_curved.get(sid, float("nan"))
        flag = "yes" if cur < lin else "no"
        comparison_lines.append("  {:<12s}  {:>16.3f}  {:>16.3f}  {}".format(sid, lin, cur, flag))

    # Time offsets: PyLIG's estimate, the curved-fit correction, and the total now applied
    offset_lines = [
        "",
        "Time offsets (s): one per camera, reference camera {} fixed, bounded to +/-{} s".format(
            info["toffset_ref_station"], args.toffset_bound),
        "{:<14s}  {:<10s}  {:>10s}  {:>12s}  {:>10s}".format("Station", "Camera", "PyLIG", "Correction",
                                                             "Total"),
    ]
    for obs, t_pylig, dt in zip(traj.observations, traj.time_diffs_final, dts):
        offset_lines.append("  {:<12s}  {:<10s}  {:>+10.4f}  {:>+12.4f}  {:>+10.4f}".format(
            str(obs.station_id), cameraID(obs.station_id), t_pylig, dt, t_pylig + dt))
    comparison_lines += offset_lines
    comparison_str = "\n".join(comparison_lines)
    print(comparison_str)
    with open(os.path.join(dir_path, prefix + "_linear_vs_curved.txt"), "w") as f:
        f.write(comparison_str + "\n")

    # --- Curved trajectory outputs ---
    traj_c.file_name = prefix
    traj_c.plot_file_type = args.imgformat
    traj_c.curved_fit = {
        "state_epoch_jd": t0_jd, "r0": r0, "v0": v0, "bstar": b0, "bstar_initial": bstar0,
        "bstar_tle": bstar_tle, "bstar_frac": args.bstar_frac, "ci_scale": args.ci_scale,
        "tle_constrain": args.tle_constrain, "space_weather": sw_vals,
        "toffset_corrections": {str(o.station_id): float(dt) for o, dt in zip(traj.observations, dts)}, **info,
    }

    saveCurvedPlots(traj_c, dir_path, prefix, args.imgformat)

    report_name = prefix + "_report.txt"
    traj_c.saveReport(dir_path, report_name, verbose=False, save_results=True)
    with open(os.path.join(dir_path, report_name), "a") as f:
        f.write("\n\nCurved fit (Keplerian + J2 + NRLMSISE-00 drag, constant B*)\n" + "-"*60 + "\n")
        f.write("State epoch        = {} UTC (JD {:.8f})\n".format(state_epoch_dt, t0_jd))
        f.write("B*                 = {:.6e} 1/m ({:.6e} 1/ER)\n".format(b0, b0*TLE_ER_M))
        f.write("B* initial         = {:.6e} 1/m, bounds +/-{:.0f}%\n".format(bstar0, 100*args.bstar_frac))
        f.write("State bounds       = {}x linear MC {}% confidence intervals{}\n".format(
            args.ci_scale, traj.uncertainties.ci, " (extended past the linear solution by 10% of the width: {})".format(
                ", ".join(info["ci_widened"])) if info["ci_widened"] else ""))
        for n, l, h in zip(["x", "y", "z", "vx", "vy", "vz"], info["bounds_lo"][:6], info["bounds_hi"][:6]):
            f.write("  {:<3s} [{:.2f}, {:.2f}]\n".format(n, l, h))
        f.write("Params at bound    = {}\n".format(", ".join(info["at_bound"]) or "none"))
        f.write("\n".join(offset_lines[1:]) + "\n")
        f.write("  (the time offsets reported above by PyLIG's report section are the totals)\n")
        if info["dh_tle_km"] is not None:
            f.write("Alt - TLE alt      = {:.2f} km at TLE epoch {} UTC ({})\n".format(
                info["dh_tle_km"], jd2Date(tle_info[0], dt_obj=True),
                "constrained to +/-{:.0f} km".format(args.tle_alt_tol_km) if args.tle_constrain
                else "not constrained"))
            f.write("  (space weather held at event-day values during back-propagation)\n")
        f.write("Fit stage {} status {} ({}), cost {:.4e}, nfev {}\n".format(
            info["stage"], info["status"], info["message"], info["cost"], info["nfev"]))

    savePickle(traj_c, dir_path, prefix + "_trajectory.pickle")
    print("Curved trajectory outputs saved to:", dir_path)
