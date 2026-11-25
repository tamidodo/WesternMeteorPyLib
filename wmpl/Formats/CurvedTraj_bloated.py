import copy
import os
import glob
from wmpl.Utils.Plotting import savePlot
from wmpl.Utils.Pickling import savePickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, least_squares
from scipy.integrate import solve_ivp
import numpy as np
import spaceweather as sw
import datetime as dt_module
from datetime import timedelta
from collections import defaultdict
from wmpl.Utils.TrajConversions import (
    cartesian2Geo,
    eci2RaDec,
    raDec2AltAz,
    raDec2ECI,
    jd2Date,
    altAz2RADec_vect,
    equatorialCoordPrecession_vect,
    J2000_JD,
    datetime2JD,
)
from wmpl.Utils.PlotMap import GroundMap
from wmpl.PythonNRLMSISE00.nrlmsise_00 import gtd7
from wmpl.PythonNRLMSISE00.nrlmsise_00_header import (
    lstCalc,
    nrlmsise_flags,
    nrlmsise_input,
    nrlmsise_output,
)
from wmpl.Formats.ECSV import loadECSVs

EARTH_RADIUS_KM = 6378.135
EARTH_MASS = 5.97219e24  # [kg]
NEWTON_G = 6.67430e-11  # [m^3/(kg·s^2)]
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
# SGP4 B* reference density, rho0 = 0.15696615 kg/(m^2 ER) = 2.461e-8 kg/m^3; B* must be in 1/m
RHO0 = 2.461e-8
MU_EARTH_KM3_S2 = 398600.4418
J2 = 1.08263e-3
R_E = 6.3781e6
mu = 3.986004418e14
h_ref = 93.5  # km — reference altitude for B*(h); set per event via setHRef()


def setHRef(value_km):
    """Set the B*(h) reference altitude for this event (arc midpoint)."""
    global h_ref
    h_ref = float(value_km)
    return h_ref


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


def llh_to_ecef(lat_deg, lon_deg, alt_m):
    """Geodetic lat/lon (degrees) + altitude (metres) → ECEF (metres)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_m) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_eci(r_ecef, dt_obj):
    """ECEF → ECI — just the transpose of the eci_to_ecef rotation."""
    theta = gmst_angle(dt_obj)
    rot_T = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return rot_T @ r_ecef


def eom(t, y, b0, t0_epoch=None, f107=None, f107A=None, ap_daily=None):
    r = y[:3]
    v = y[3:]
    r_mag = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])

    a = -mu / r_mag**3 * r

    z2 = r[2] ** 2
    r2 = r_mag**2
    j2_fac = (3 / 2) * J2 * mu * R_E**2 / r_mag**5
    a_j2 = j2_fac * np.array(
        [r[0] * (5 * z2 / r2 - 1), r[1] * (5 * z2 / r2 - 1), r[2] * (5 * z2 / r2 - 3)]
    )

    a_drag = np.zeros(3)
    if b0 is not None and t0_epoch is not None:
        vmag2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if vmag2 > 0.0:
            vmag = np.sqrt(vmag2)
            current_epoch = t0_epoch + timedelta(seconds=float(t))
            r_ecef = eci_to_ecef(r, current_epoch)
            geo_lat, geo_lon, alt = ecef_to_llh(r_ecef)

            drag_input = nrlmsise_input()
            drag_input.f107A = f107A
            drag_input.f107 = f107
            drag_input.ap = ap_daily
            drag_input.doy = current_epoch.timetuple().tm_yday
            drag_input.sec = (
                current_epoch.hour * 3600
                + current_epoch.minute * 60
                + current_epoch.second
                + current_epoch.microsecond / 1.0e6
            )
            drag_input.alt = alt
            drag_input.g_lat = geo_lat
            drag_input.g_long = geo_lon
            lstCalc(drag_input)

            drag_output = nrlmsise_output()
            gtd7(drag_input, _DRAG_FLAGS, drag_output)
            rho = drag_output.d[5] * 1000.0

            a_drag_mag = (rho / RHO0) * abs(b0) * vmag2
            a_drag = -a_drag_mag / vmag * v

    return np.concatenate([v, a + a_j2 + a_drag])


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


def _build_dense_trajectory(
    r0, v0, t_max, b0, t0_epoch, f107, f107A, ap_daily, n_fine=2000
):
    """
    One integration per cost-function evaluation, covering every station's
    time range. Returns a dense interpolant plus a precomputed arc-length grid.
    """
    y0 = np.concatenate([r0, v0])
    sol = solve_ivp(
        eom,
        (0.0, t_max),
        y0,
        method="DOP853",
        args=(b0, t0_epoch, f107, f107A, ap_daily),
        dense_output=True,
        rtol=1e-9,
        atol=[1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9],
    )

    t_fine = np.linspace(0.0, t_max, n_fine)
    pos_fine = sol.sol(t_fine)[:3, :].T
    diffs = np.diff(pos_fine, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc_fine = np.concatenate(([0.0], np.cumsum(seg_len)))

    return sol.sol, t_fine, arc_fine


class MinimalObs:
    """
    Lightweight drop-in for PyLIG's ObservedPoints, holding only what the
    curved solver actually reads. Used for cameras that PyLIG dropped because
    they had no temporal overlap with any other station.
    """

    def __init__(self, station_id, time_data, jd_data, stat_eci, meas_eci, abs_mags):
        self.station_id = station_id
        self.time_data = np.asarray(time_data, dtype=float)  # s, common axis
        self.JD_data = np.asarray(jd_data, dtype=float)  # absolute JD
        self.stat_eci_los = np.asarray(stat_eci, dtype=float)  # (N,3) metres
        self.meas_eci_los = np.asarray(meas_eci, dtype=float)  # (N,3) unit vecs
        self.absolute_magnitudes = np.asarray(abs_mags, dtype=float)
        self.ignore_list = np.zeros(len(time_data), dtype=int)
        self.ignore_station = False
        n = len(time_data)
        nan = np.full(n, np.nan)
        self.meas_ht    = nan.copy()
        self.model_ht   = nan.copy()
        self.lat_data   = nan.copy()
        self.lon_data   = nan.copy()
        self.range_data = nan.copy()
        self.ra_data    = nan.copy()
        self.dec_data   = nan.copy()
        self.azim_data  = nan.copy()
        self.alt_data   = nan.copy()


def loadExtraECSVsForCurvedSolver(ecsv_dir, jdt_ref):
    """
    Parse every *.ecsv in ecsv_dir into MinimalObs objects on the common
    time axis defined by jdt_ref (taken from the primary PyLIG pickle).

    Mirrors what PyLIG's loadECSVs does for coordinate conversion, but skips
    the MeteorObservation machinery and produces only what the curved solver
    needs.
    """
    ecsv_paths = sorted(glob.glob(os.path.join(ecsv_dir, "*.ecsv")))
    if not ecsv_paths:
        print(f"No ECSV files found in {ecsv_dir}")
        return []

    obs_list = []

    for ecsv_path in ecsv_paths:
        station_lat = station_lon = station_ele = station_id = None

        # ── parse header ────────────────────────────────────────────────────
        with open(ecsv_path) as f:
            for line in f:
                if not line.startswith("#"):
                    continue
                line = (
                    line.replace("\n", "")
                    .replace("\r", "")
                    .replace("{", "")
                    .replace("}", "")
                )
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0]
                val = parts[1]
                if "obs_latitude" in key:
                    station_lat = float(val)
                if "obs_longitude" in key:
                    station_lon = float(val)
                if "obs_elevation" in key:
                    station_ele = float(val)
                if "camera_id" in key:
                    station_id = val.strip().strip("'")
                if station_id is None and "dfn_camera_codename" in key:
                    station_id = val.strip().strip("'")

        if any(v is None for v in [station_lat, station_lon, station_ele, station_id]):
            print(f"Skipping {ecsv_path}: incomplete header")
            continue

        # ── parse data ──────────────────────────────────────────────────────
        data = np.loadtxt(ecsv_path, comments="#", delimiter=",", dtype=str)
        header = data[0].tolist()
        data = data[1:]

        if len(data) < 4:
            print(f"Skipping {station_id}: fewer than 4 points")
            continue

        dt_indx = header.index("datetime")
        azim_indx = header.index("azimuth")
        alt_indx = header.index("altitude")
        mag_indx = header.index("mag_data")

        azim_deg = data[:, azim_indx].astype(float)
        alt_deg = data[:, alt_indx].astype(float)
        mag_data = data[:, mag_indx].astype(float)

        jd_data = np.array(
            [
                datetime2JD(dt_module.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f"))
                for s in data[:, dt_indx]
            ]
        )

        # Time on the common axis (seconds since the primary pickle's jdt_ref)
        time_data = (jd_data - jdt_ref) * 86400.0

        # ── az/alt → RA/Dec → precess to J2000 (same as PyLIG's loadECSVs) ─
        ra_data, dec_data = altAz2RADec_vect(
            np.radians(azim_deg),
            np.radians(alt_deg),
            jd_data,
            np.radians(station_lat),
            np.radians(station_lon),
        )
        ra_data, dec_data = equatorialCoordPrecession_vect(
            jd_data[0], J2000_JD.days, ra_data, dec_data
        )

        # ── LoS unit vectors in ECI ─────────────────────────────────────────
        meas_eci = np.array([raDec2ECI(ra, dec) for ra, dec in zip(ra_data, dec_data)])

        # ── station ECI position at each timestamp ──────────────────────────
        # Station is fixed on the ground; rotate its ECEF position to ECI at
        # each measurement epoch using the GMST angle.
        stat_ecef = llh_to_ecef(station_lat, station_lon, station_ele)
        stat_eci = np.array(
            [ecef_to_eci(stat_ecef, jd2Date(jd, dt_obj=True)) for jd in jd_data]
        )

        obs_list.append(MinimalObs(station_id, time_data, jd_data, stat_eci, meas_eci, mag_data))
        print(
            f"  Loaded extra station {station_id}: {len(time_data)} points, "
            f"t_rel=[{time_data[0]:.2f}, {time_data[-1]:.2f}] s"
        )

    return obs_list


def angleSumOrbitalArc(
    params,
    observations,
    v_expected,
    v0_unit_guess,
    t0_global,
    t0_epoch,
    f107,
    f107A,
    ap_daily,
    obs_group_sizes,
    sigma_ang_arcsec=60.0,
    sigma_v=50.0,
    sigma_dir_deg=0.5,
    max_fit_seconds=None,
):
    r0 = params[:3]
    v0 = params[3:6]
    b0 = params[6]

    t_max = max(obs.time_data[-1] - t0_global for obs in observations)
    dense_sol, t_fine, arc_fine = _build_dense_trajectory(
        r0, v0, t_max, b0, t0_epoch, f107, f107A, ap_daily
    )

    def _station_angle_cost(obs_list):

        # Group by station_id within this group — all FF files from the
        # same physical camera share the same ID here, so this correctly
        # pools NZ002A's multiple FF files into one cost value, while
        # keeping NZ000D and NZ000D_2 as separate cameras (different IDs)
        by_sid = defaultdict(list)
        for obs in obs_list:
            by_sid[str(obs.station_id)].append(obs)

        sigma_ang = np.radians(sigma_ang_arcsec/3600.0)

        chi2_sum = 0.0
        n_pts = 0
        for sid, sid_obs_list in by_sid.items():
            for obs in sid_obs_list:
                t_rel = obs.time_data - t0_global
                if len(t_rel) < 2:
                    continue
                traj_eci = dense_sol(t_rel)[:3, :].T
                mask = obs.ignore_list == 0
                if max_fit_seconds is not None:
                    mask &= t_rel <= max_fit_seconds
                if not np.any(mask):
                    continue
                model_rays = traj_eci[mask] - obs.stat_eci_los[mask]
                norms = np.linalg.norm(model_rays, axis=1)
                model_rays_unit = model_rays / norms[:, None]
                cos_angles = np.clip(
                    np.einsum("ij,ij->i", obs.meas_eci_los[mask],
                            model_rays_unit), -1, 1)
                angles = np.arccos(cos_angles)
                chi2_sum += float(np.sum((angles/sigma_ang)**2))
                n_pts += int(np.sum(mask))

        if n_pts == 0:
            return None
        return chi2_sum

    # Slice the combined observation list by group
    groups = []
    start = 0
    for size in obs_group_sizes:
        groups.append(observations[start : start + size])
        start += size
    if start < len(observations):  # any remainder not covered by group sizes
        groups.append(observations[start:])
    group_costs = [
        c for g in groups if g for c in [_station_angle_cost(g)] if c is not None
    ]
    angle_cost = float(np.sum(group_costs)) if group_costs else 0.0

    v_mag = np.linalg.norm(v0)
    speed_penalty = ((v_mag - v_expected) / sigma_v) ** 2
    v_unit = v0 / v_mag
    cos_dir = np.clip(np.dot(v_unit, v0_unit_guess), -1, 1)
    direction_penalty = (np.degrees(np.arccos(cos_dir)) / sigma_dir_deg) ** 2

    if angleSumOrbitalArc.verbose:
        print(f"    angle={angle_cost:.4e}  speed={speed_penalty:.4e}  "
              f"dir={direction_penalty:.4e}")

    return angle_cost + speed_penalty + direction_penalty

angleSumOrbitalArc.verbose = False


def computeAngularRMSD(
    observations,
    r0,
    v0,
    label="",
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

    print(f"\n{label} — per-station angular RMSD (arcsec):")
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

        rays = pos - obs.stat_eci_los[m]
        rays = rays/np.linalg.norm(rays, axis=1)[:, None]
        cos_a = np.clip(np.einsum("ij,ij->i", obs.meas_eci_los[m], rays), -1, 1)
        results[str(obs.station_id)] = float(
            np.sqrt(np.mean((np.degrees(np.arccos(cos_a))*3600.0)**2)))

    print(f"\n{label} — per-station angular RMSD (arcsec):")
    for sid, rmsd in sorted(results.items()):
        print(f"  {sid:12s}: {rmsd:.3f} arcsec")

    return results


def get_space_weather(epoch_datetime):
    """
    Return (f107_previous_day, f107_81day_avg, ap_daily) for a given datetime.

    NRLMSISE-00 inputs:
      - f107:  F10.7 observed for the PREVIOUS day (not the day of the event)
      - f107A: 81-day average of F10.7 centred on the day of the event
      - ap:    daily Ap index for the day of the event
    """
    df = sw.sw_daily(
        swpath_5y="/Users/tammydo/Documents/phd-uwo/space_debris/spaceweather5Y.txt",
        update=False,
    )

    date_str = epoch_datetime.strftime("%Y-%m-%d")
    prev_date_str = (epoch_datetime - timedelta(days=1)).strftime("%Y-%m-%d")

    row_today = df.loc[date_str]
    row_prev = df.loc[prev_date_str]

    f107_prev_day = float(row_prev["f107_obs"])  # previous day observed F10.7
    f107_81day_avg = float(row_today["f107_81ctr_obs"])  # 81-day centred average
    ap_daily = float(row_today["Apavg"])  # daily mean Ap

    return f107_prev_day, f107_81day_avg, ap_daily


def angleResidualsOrbitalArc(
    params, observations, t0_global, t0_epoch, f107, f107A, ap_daily,
    sigma_ang_arcsec=60.0, max_fit_seconds=None,
):
    """Per-point normalised angular residuals, for scipy.optimize.least_squares."""
    r0, v0, b0 = params[:3], params[3:6], params[6]

    t_max = max(obs.time_data[-1] - t0_global for obs in observations)
    dense_sol, _, _ = _build_dense_trajectory(
        r0, v0, t_max, b0, t0_epoch, f107, f107A, ap_daily
    )

    sigma_ang = np.radians(sigma_ang_arcsec/3600.0)
    res = []
    for obs in observations:
        t_rel = obs.time_data - t0_global
        if len(t_rel) < 2:
            continue
        mask = obs.ignore_list == 0
        if max_fit_seconds is not None:
            mask &= t_rel <= max_fit_seconds
        if not np.any(mask):
            continue
        traj_eci = dense_sol(t_rel)[:3, :].T
        rays = traj_eci[mask] - obs.stat_eci_los[mask]
        rays = rays/np.linalg.norm(rays, axis=1)[:, None]
        cos_a = np.clip(np.einsum("ij,ij->i", obs.meas_eci_los[mask], rays), -1, 1)
        res.append(np.arccos(cos_a)/sigma_ang)

    return np.concatenate(res) if res else np.zeros(1)


def fitSatelliteTrajectory(
    observations,
    traj_los,
    jdt_ref,
    bstar_guess=1e-4,
    obs_group_sizes=None,
    max_fit_sec=None,
    v_init_override=None,
):
    """
    traj_los: result from the existing LoS solver (for initial guess)
    """
    if obs_group_sizes is None:
        obs_group_sizes = [len(observations)]
    r0_guess = traj_los.state_vect_mini
    v_init_use = traj_los.v_init if v_init_override is None else float(v_init_override)
    v0_guess = -traj_los.radiant_eci_mini * v_init_use

    t0_global = min(obs.time_data[0] for obs in observations)
    t0_epoch = jd2Date(jdt_ref, dt_obj=True) + timedelta(seconds=float(t0_global))
    f107, f107A, ap_daily = get_space_weather(t0_epoch)
    # Tight bounds: positions within ~50 km, velocities within ~500 m/s of guess

    def _optimization_callback(xk):
        _optimization_callback.count += 1
        if _optimization_callback.count % 100 == 0:
            print(
                f"  Iteration {_optimization_callback.count}, "
                f"current params: b0={xk[6]:.4e}"
            )

    _optimization_callback.count = 0

    x0_ls = np.concatenate([r0_guess, v0_guess, [bstar_guess]])

    x_scale_affine = np.array([
        1.0e3, 1.0e3, 1.0e3,          # position: 1 km per unit
        1.0e1, 1.0e1, 1.0e1,          # velocity: 10 m/s per unit
        max(abs(bstar_guess), 1e-9),  # b0: its own magnitude per unit
    ])

    def _expand(p):
        return x0_ls + (p - 1.0)*x_scale_affine

    def _resid_scaled(p):
        return angleResidualsOrbitalArc(
            _expand(p), observations, t0_global, t0_epoch, f107, f107A, ap_daily,
            60.0, max_fit_sec)

    lo = np.full(7, -np.inf)
    lo[6] = 1.0 - x0_ls[6]/x_scale_affine[6]      # keeps b0 >= 0
    x_start = np.ones(7)
    result = None

    for attempt, ds in enumerate([0.5, 0.1, 0.05, 0.02]):
        res = least_squares(
            _resid_scaled, x_start,
            bounds=(lo, np.full(7, np.inf)),
            diff_step=ds, x_scale="jac", tr_solver="exact",
            loss="linear", f_scale=10.0,
            method="trf", xtol=1e-8, ftol=1e-8, gtol=1e-8,
            max_nfev=2000, verbose=2,
        )
        print(f"  attempt {attempt}: cost {res.cost:.4e}  "
              f"optimality {res.optimality:.2e}  nfev {res.nfev}  "
              f"status {res.status}")
        if result is None or res.cost < result.cost:
            result = res
            x_start = res.x
        if res.status in (2, 3) and res.optimality < 1e3:
            break

    if result.optimality > 1e3:
        print(f"  WARNING: optimality {result.optimality:.2e} after "
              f"{result.nfev} evaluations — trust region collapsed, "
              f"not converged.")

    x_fit = _expand(result.x)
    r0_fit = x_fit[:3]
    v0_fit = x_fit[3:6]
    b0_fit = x_fit[6]

    print(f"  b0 = {b0_fit:.4e}")
    return r0_fit, v0_fit, b0_fit, t0_epoch, f107, f107A, ap_daily


def computeCurvedTrajectoryMetrics(
    observations, r0_fit, v0_fit, b0_fit, t0_epoch, f107, f107A, ap_daily, jdt_ref
):
    t0_global = min(obs.time_data[0] for obs in observations)
    t_max_global = max(obs.time_data[-1] for obs in observations) - t0_global

    t_fine = np.linspace(0.0, t_max_global, 5000)
    pos_fine = keplerian_propagator(
        r0_fit, v0_fit, t_fine, b0_fit, t0_epoch, f107, f107A, ap_daily
    )
    diffs = np.diff(pos_fine, axis=0)
    arc_fine = np.concatenate(([0.0], np.cumsum(np.linalg.norm(diffs, axis=1))))

    by_sid = defaultdict(list)
    for obs in observations:
        by_sid[str(obs.station_id)].append(obs)

    merged_observations = []
    for sid, obs_list in by_sid.items():
        if len(obs_list) == 1:
            merged_observations.append(obs_list[0])
            continue

        # sort by start time, then concatenate all per-point arrays
        obs_list_sorted = sorted(obs_list, key=lambda o: o.time_data[0])
        ref = obs_list_sorted[0]

        class MergedObs:
            pass

        m = MergedObs()
        m.station_id   = ref.station_id
        m.ignore_station = ref.ignore_station
        m.time_data    = np.concatenate([o.time_data    for o in obs_list_sorted])
        m.JD_data      = np.concatenate([o.JD_data      for o in obs_list_sorted])
        m.stat_eci_los = np.concatenate([o.stat_eci_los for o in obs_list_sorted])
        m.meas_eci_los = np.concatenate([o.meas_eci_los for o in obs_list_sorted])
        m.ignore_list  = np.concatenate([o.ignore_list  for o in obs_list_sorted])
        m.absolute_magnitudes = np.concatenate([
            getattr(o, "absolute_magnitudes", np.full(len(o.time_data), np.nan))
            for o in obs_list_sorted])
        merged_observations.append(m)

    results = []
    for obs in observations:
        t_rel = obs.time_data - t0_global
        jd_corr = jdt_ref + obs.time_data / 86400.0

        sol = solve_ivp(
            eom,
            (0.0, t_rel[-1]),
            np.concatenate([r0_fit, v0_fit]),
            t_eval=t_rel,
            method="DOP853",
            args=(b0_fit, t0_epoch, f107, f107A, ap_daily),
            rtol=1e-9,
            atol=[1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6],
        )

        pos_model = sol.y[:3, :].T  # (N, 3) metres
        vel_model = sol.y[3:, :].T  # (N, 3) m/s

        # --- Arc length from first propagated point ---
        arc_lengths = np.interp(t_rel, t_fine, arc_fine)

        # --- Heights at model positions ---
        model_ht = np.zeros(len(t_rel))
        model_lat  = np.zeros(len(t_rel))
        model_lon  = np.zeros(len(t_rel))
        model_ht_m = np.zeros(len(t_rel))   # metres, not km
        range_data = np.zeros(len(t_rel))
        ra_obs     = np.zeros(len(t_rel))
        dec_obs    = np.zeros(len(t_rel))
        ra_line    = np.zeros(len(t_rel))
        dec_line   = np.zeros(len(t_rel))
        azim_obs   = np.zeros(len(t_rel))
        alt_obs    = np.zeros(len(t_rel))
        azim_line  = np.zeros(len(t_rel))
        alt_line   = np.zeros(len(t_rel))

        for k in range(len(t_rel)):
            jd_k   = jd_corr[k]
            pos_k  = pos_model[k]
            stat_k = obs.stat_eci_los[k]
            meas_k = obs.meas_eci_los[k]   # observed LoS unit vector

            # geodetic position of the model point
            lat_k, lon_k, ht_km = cartesian2Geo(jd_k, *pos_k)
            model_lat[k]  = lat_k
            model_lon[k]  = lon_k
            model_ht[k]   = ht_km * 1000.0   # keep existing metres convention
            model_ht_m[k] = ht_km * 1000.0

            # range from station to model position
            range_data[k] = np.linalg.norm(pos_k - stat_k)

            # RA/Dec of the OBSERVED LoS (what the camera measured)
            ra_o, dec_o = eci2RaDec(meas_k)
            ra_obs[k]  = np.degrees(ra_o)
            dec_obs[k] = np.degrees(dec_o)

            # az/alt of the observed LoS at the station's geodetic position
            stat_lat_k, stat_lon_k, _ = cartesian2Geo(jd_k, *stat_k)
            az_o, al_o = raDec2AltAz(ra_o, dec_o, jd_k, stat_lat_k, stat_lon_k)
            azim_obs[k] = np.degrees(az_o)
            alt_obs[k]  = np.degrees(al_o)

            # RA/Dec and az/alt of the MODEL LoS (direction from station to model point)
            model_los = pos_k - stat_k
            model_los_unit = model_los / np.linalg.norm(model_los)
            ra_l, dec_l = eci2RaDec(model_los_unit)
            ra_line[k]  = np.degrees(ra_l)
            dec_line[k] = np.degrees(dec_l)
            az_l, al_l = raDec2AltAz(ra_l, dec_l, jd_k, stat_lat_k, stat_lon_k)
            azim_line[k] = np.degrees(az_l)
            alt_line[k]  = np.degrees(al_l)

        # --- Spatial residuals (metres, PyLIG sign convention) ---
        h_residuals = np.zeros(len(t_rel))
        v_residuals = np.zeros(len(t_rel))

        for j in range(len(t_rel)):

            stat_j = obs.stat_eci_los[j]  # station ECI (m)
            meas_j = obs.meas_eci_los[j]  # unit LoS direction (observed)
            pos_j = pos_model[j]  # model satellite position (m)
            vel_j = vel_model[j]  # model velocity (m/s)

            # Point on the observed LoS closest to the model position
            # (equivalent to the obs_cpa in calcSpatialResidual)
            lam = np.dot(pos_j - stat_j, meas_j)  # range to CPA (m)
            obs_cpa = stat_j + lam * meas_j  # obs LoS CPA point

            # Displacement: from model point to observed CPA point
            # (same as p = obs_cpa - rad_cpa in PyLIG, but rad_cpa = pos_j here)
            p = obs_cpa - pos_j

            # --- Local reference frame at this model point ---
            # Reproduce PyLIG's frame from calcSpatialResidual:
            #   x-axis → along-track (velocity direction replaces radiant direction)
            #   y-axis → horizontal, 90° CCW of x along horizon
            #   z-axis → up from trajectory (vz = hx × hy)
            #
            # PyLIG derives this in ECI via azimuth/elevation of the radiant;
            # for the curved model we use the velocity unit vector instead.

            jd_j = jd_corr[j]
            lat_j, lon_j, _ = cartesian2Geo(jd_j, *pos_j)

            vel_norm = np.linalg.norm(vel_j)
            if vel_norm < 1.0:
                continue
            vel_unit = vel_j / vel_norm

            # RA/Dec of the velocity direction → azimuth/elevation at model point
            rad_ra, rad_dec = eci2RaDec(-vel_unit)
            rad_az, rad_el = raDec2AltAz(rad_ra, rad_dec, jd_j, lat_j, lon_j)

            if np.abs(np.pi / 2.0 - rad_el) <= 1e-12:
                rad_az = 0.0

            # y-axis: 90° CCW of x along horizon (PyLIG: azimy = azimx - pi/2)
            azimy = (rad_az - np.pi / 2.0) % (2 * np.pi)
            elevy = 0.0

            # z-axis: behind you looking along velocity (PyLIG: azimz = azimx + pi)
            azimz = (rad_az + np.pi) % (2 * np.pi)
            elevz = np.pi / 2.0 - rad_el

            ray_j, decj = altAz2RADec_vect(azimy, elevy, jd_j, lat_j, lon_j)
            raz, decz = altAz2RADec_vect(azimz, elevz, jd_j, lat_j, lon_j)

            hy_eci = raDec2ECI(ray_j, decj)  # horizontal cross-track unit vector
            vz_eci = raDec2ECI(raz, decz)  # vertical unit vector

            # Project displacement onto h and v axes (same as PyLIG equations)
            # hres = sign(ehy) * hypot(ehx, ehy)  but ehx ≈ 0 (p ⊥ x-axis)
            ehx = np.dot(p, vel_unit)  # along-track component (≈ 0)
            ehy = np.dot(p, hy_eci)

            h_residuals[j] = np.sign(ehy) * np.hypot(ehx, ehy)
            v_residuals[j] = np.dot(p, vz_eci)

        # Zero out ignored points so RMS is computed only on used data
        mask = obs.ignore_list == 0
        h_res_rms = np.sqrt(np.mean(h_residuals[mask] ** 2))
        v_res_rms = np.sqrt(np.mean(v_residuals[mask] ** 2))

        results.append({
            "station_id":          obs.station_id,
            "time_data":           obs.time_data,
            "JD_data":             jd_corr,
            "stat_eci_los":        obs.stat_eci_los,
            "meas_eci_los":        obs.meas_eci_los,
            "ignore_list":         obs.ignore_list,
            "absolute_magnitudes": obs.absolute_magnitudes,
            "model_lat":           model_lat,
            "model_lon":           model_lon,
            "model_ht":            model_ht,
            "range_data":          range_data,
            "ra_obs":              ra_obs,
            "dec_obs":             dec_obs,
            "ra_line":             ra_line,
            "dec_line":            dec_line,
            "azim_obs":            azim_obs,
            "alt_obs":             alt_obs,
            "azim_line":           azim_line,
            "alt_line":            alt_line,
            "h_residuals":         h_residuals,
            "v_residuals":         v_residuals,
            "h_res_rms":           h_res_rms,
            "v_res_rms":           v_res_rms,
            "state_vect_dist":     arc_lengths,
            "rbeg_ele":            model_ht[mask][0] if mask.any() else model_ht[0],
        })

    return results


def _lineFunc(t, v, c):
    return v * t + c


def plotCurvedGroundTrack(traj_c, curved_metrics, output_dir=None, show_plot=True, plot_file_type='png'):
    """
    Generate a ground track map from curved trajectory metrics.
    Uses model-derived lat/lon positions rather than PyLIG's geometric intersections.
    """

    metric_by_sid = {str(m["station_id"]): m for m in curved_metrics}

    # ── collect all lat/lon needed to set map extent ──────────────────────────
    all_lat, all_lon = [], []

    # station positions from observations
    for obs in traj_c.observations:
        if hasattr(obs, 'lat') and hasattr(obs, 'lon'):
            all_lat.append(obs.lat)
            all_lon.append(obs.lon)

    # model track points from metrics
    for m in curved_metrics:
        mask = m["ignore_list"] == 0
        all_lat.extend(m["model_lat"][mask].tolist())
        all_lon.extend(m["model_lon"][mask].tolist())

    # begin/end points
    for attr in ['rbeg_lat', 'rend_lat']:
        if hasattr(traj_c, attr):
            all_lat.append(getattr(traj_c, attr))
    for attr in ['rbeg_lon', 'rend_lon']:
        if hasattr(traj_c, attr):
            all_lon.append(getattr(traj_c, attr))

    if not all_lat:
        print("No lat/lon data available for ground track plot.")
        return

    # ── init map ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gmap = GroundMap(all_lat, all_lon, border_size=50,
                     color_scheme='light', ax=ax)

    # ── plot model track per station ──────────────────────────────────────────
    colors  = plt.cm.viridis(np.linspace(0, 0.85, len(curved_metrics)))
    markers = [('x', 1), ('+', 1.5), ('o', 0.8), ('s', 0.8),
               ('^', 0.8), ('D', 0.8)]
    plotted_stations = []

    for i, m in enumerate(sorted(curved_metrics,
                                  key=lambda x: np.min(x["model_ht"]),
                                  reverse=True)):
        sid  = str(m["station_id"])
        mask = m["ignore_list"] == 0
        lat  = m["model_lat"]
        lon  = m["model_lon"]
        color = colors[i % len(colors)]
        marker, sm = markers[i % len(markers)]

        # model track points
        if np.any(mask):
            gmap.plot(lat[mask], lon[mask], c=color, linewidth=1.0,
                      label=sid)

        # ignored points
        if np.any(~mask):
            gmap.scatter(lat[~mask], lon[~mask], c='k', marker='x', s=5, alpha=0.5)

        # station position
        obs_match = next((obs for obs in traj_c.observations
                          if str(obs.station_id) == sid
                          and hasattr(obs, 'lat')), None)
        if obs_match is not None and sid not in plotted_stations:
            gmap.scatter([obs_match.lat], [obs_match.lon],
                         s=sm * 15, marker=marker, c=[color])
            plotted_stations.append(sid)

    # ── begin/end markers ────────────────────────────────────────────────────
    t0_global = min(obs.time_data[0] for obs in traj_c.observations
                if str(obs.station_id) in metric_by_sid)
    t_end = max(obs.time_data[-1] for obs in traj_c.observations
                if str(obs.station_id) in metric_by_sid) - t0_global
    pos_end = keplerian_propagator(
        r0_fit, v0_fit, np.array([t_end]),
        b0_fit, t0_epoch, f107, f107A, ap_daily)[0]

    jd_end = traj.jdt_ref + (t0_global + t_end) / 86400.0
    lat_end, lon_end, ht_end = cartesian2Geo(jd_end, *pos_end)

    traj_c.rend_lat = lat_end
    traj_c.rend_lon = lon_end
    traj_c.rend_ele = ht_end * 1000.0

    if hasattr(traj_c, 'rbeg_lat'):
        gmap.scatter([traj_c.rbeg_lat], [traj_c.rbeg_lon],
                     c='lime', marker='*', s=80, label='Begin')
    if hasattr(traj_c, 'rend_lat'):
        gmap.scatter([traj_c.rend_lat], [traj_c.rend_lon],
                     c='red', marker='*', s=80, label='End')

    legend_size = 5 if len(curved_metrics) >= 10 else 7
    ax.legend(loc='upper left', prop={'size': legend_size})
    ax.set_title("Curved trajectory ground track")

    plt.tight_layout()

    if output_dir:
        savePlot(plt, f"{traj_c.file_name}_curved_ground_track.{plot_file_type}", output_dir)

    if show_plot:
        plt.show()
    else:
        plt.clf()
        plt.close()


def saveCurvedTrajectory(
    traj,
    r0_fit,
    v0_fit,
    curved_metrics,
    dir_path,
    all_observations=None,
    b0_fit=0.0,
    t0_epoch=None,
    f107=None,
    f107A=None,
    ap_daily=None,
    show_plots=True,
    plot_file_type="png",
):
    """
    Save curved-trajectory diagnostic outputs in the same format as Trajectory.savePlots /
    saveReport / savePickle.  A deep copy of `traj` is patched with curved-model residuals,
    lags, velocities and angular residuals, then the standard Trajectory methods are called so
    the file names and formats are identical to those produced by solveTrajectoryGeneric.

    Output files (all written to dir_path, prefixed with traj.file_name + '_curved'):
        _curved_trajectory.pickle
        _curved_report.txt
        _curved_all_angular_residuals.<ext>
        _curved_all_spatial_total_residuals_height.<ext>
        _curved_lags_all.<ext>
        _curved_lengths.<ext>
        _curved_velocities.<ext>
        _curved_<station_id>_spatial_residuals.<ext>  (one per station)
    """

    traj_c = copy.deepcopy(traj)
    traj_c.plot_file_type = plot_file_type
    traj_c.save_results = True
    traj_c.show_jacchia = (
        False  # Jacchia (ablation) model doesn't apply to orbital arcs
    )
    traj_c.show_plots = show_plots
    traj_c.plot_all_spatial_residuals = False

    if all_observations is not None:
        existing_sids = {str(obs.station_id) for obs in traj_c.observations}
    for obs in all_observations:
        sid = str(obs.station_id)
        if sid not in existing_sids:
            traj_c.observations.append(copy.deepcopy(obs))
            existing_sids.add(sid)

    # Map curved metrics by station_id for O(1) lookup
    metric_by_sid = {str(m["station_id"]): m for m in curved_metrics}

    t0_global = min(obs.time_data[0] for obs in traj_c.observations)

    # ── 1. Update per-observation attributes ─────────────────────────────────────
    for obs in traj_c.observations:
        sid = str(obs.station_id)
        m = metric_by_sid.get(sid)
        if m is None:
            continue

        obs.state_vect_dist = m["state_vect_dist"]
        obs.time_data = m["time_data"] 
        obs.stat_eci_los = m["stat_eci_los"]
        obs.meas_eci_los = m["meas_eci_los"]
        obs.absolute_magnitudes = m["absolute_magnitudes"]
        obs.ignore_list = m["ignore_list"]
        obs.model_ht = m["model_ht"]
        obs.meas_ht = m["model_ht"]
        obs.h_residuals = m["h_residuals"]
        obs.v_residuals = m["v_residuals"]
        obs.h_res_rms = m["h_res_rms"]
        obs.v_res_rms = m["v_res_rms"]
        obs.rbeg_ele = m["rbeg_ele"]
        obs.JD_data          = m["JD_data"]
        obs.lat_data         = m["model_lat"]
        obs.lon_data         = m["model_lon"]
        obs.range_data       = m["range_data"]
        obs.ra_data          = m["ra_obs"]
        obs.dec_data         = m["dec_obs"]
        obs.azim_data        = m["azim_obs"]
        obs.alt_data         = m["alt_obs"]
        obs.ra_line          = m["ra_line"]
        obs.dec_line         = m["dec_line"]
        obs.azim_line        = m["azim_line"]
        obs.alt_line         = m["alt_line"]

        # obs.length is the arc distance from this station's FIRST point (used in saveReport)
        obs.length = obs.state_vect_dist - obs.state_vect_dist[0]

        # Angular residuals: angle between observed LoS and curved-model LoS
        t_rel = obs.time_data - t0_global
        traj_eci = keplerian_propagator(
            r0_fit, v0_fit, t_rel, b0_fit, t0_epoch, f107, f107A, ap_daily
        )
        model_rays = traj_eci - obs.stat_eci_los
        norms = np.linalg.norm(model_rays, axis=1)
        valid = norms >= 1.0
        ang_res = np.zeros(len(t_rel))
        model_rays_unit = np.where(valid[:, None], model_rays / norms[:, None], 0.0)
        cos_a = np.clip(
            np.einsum("ij,ij->i", obs.meas_eci_los, model_rays_unit), -1.0, 1.0
        )
        ang_res[valid] = np.arccos(cos_a[valid])

        obs.ang_res = ang_res
        mask = obs.ignore_list == 0
        obs.ang_res_std = (
            float(np.sqrt(np.mean(ang_res[mask] ** 2))) if np.any(mask) else 0.0
        )

        # Point-to-point velocities from curved arc (same formula as Trajectory.py)
        svd = obs.state_vect_dist
        dists_shifted = np.r_[0.0, svd][:-1]
        dists_diffs = svd - dists_shifted
        time_shifted = np.r_[0.0, obs.time_data][:-1]
        time_diffs = obs.time_data - time_shifted
        time_diffs[time_diffs == 0] = np.finfo(np.float64).eps
        obs.velocities = dists_diffs / time_diffs
        obs.velocities_prev_point = obs.velocities.copy()

    # ── 2. Global linear velocity fit across all curved arcs ─────────────────────
    all_t, all_svd = [], []
    for obs in traj_c.observations:
        sid = str(obs.station_id)
        if sid not in metric_by_sid:
            continue
        mask = obs.ignore_list == 0
        all_t.extend(obs.time_data[mask].tolist())
        all_svd.extend(obs.state_vect_dist[mask].tolist())
    all_t = np.array(all_t)
    all_svd = np.array(all_svd)
    sort_idx = np.argsort(all_t)
    n_fit = max(4, int(0.4 * len(all_t)))
    try:
        velocity_fit, _ = curve_fit(
            _lineFunc,
            all_t[sort_idx][:n_fit],
            all_svd[sort_idx][:n_fit],
            p0=[np.linalg.norm(v0_fit), 0.0],
        )
    except Exception:
        velocity_fit = np.array([np.linalg.norm(v0_fit), 0.0])
    traj_c.velocity_fit = velocity_fit

    # ── 3. Lag = arc distance minus global linear velocity fit ───────────────────
    for obs in traj_c.observations:
        if str(obs.station_id) not in metric_by_sid:
            continue
        obs.lag_line = list(velocity_fit)
        obs.lag = obs.state_vect_dist - _lineFunc(obs.time_data, *velocity_fit)

    # ── 4. Traj-level state vector ───────────────────────────────────────────────
    traj_c.state_vect_mini = r0_fit
    v0_norm = np.linalg.norm(v0_fit)
    traj_c.radiant_eci_mini = -v0_fit / v0_norm
    traj_c.v_init = v0_norm

    # ── 5. Save plots ────────────────────────────────────────────────────────────
    plotCurvedGroundTrack(
        traj_c, curved_metrics,
        output_dir=dir_path,
        show_plot=show_plots,
        plot_file_type=plot_file_type
    )
    file_name = traj_c.file_name + "_curved"
    traj_c.savePlots(dir_path, file_name, show_plots=show_plots)

    # ── 6. Save report ───────────────────────────────────────────────────────────
    try:
        traj_report = copy.deepcopy(traj)   # fresh copy — original unpatched observations
        traj_report.state_vect_mini  = r0_fit
        traj_report.radiant_eci_mini = -v0_fit / np.linalg.norm(v0_fit)
        traj_report.v_init           = np.linalg.norm(v0_fit)
        traj_report.velocity_fit     = velocity_fit
        traj_report.plot_file_type   = plot_file_type
        traj_report.save_results     = True
        # patch per-obs lag/velocity only on the observations that exist
        # in the original traj (no MinimalObs, original array lengths intact)
        metric_by_sid_report = {str(m["station_id"]): m for m in curved_metrics}
        for obs in traj_report.observations:
            m = metric_by_sid_report.get(str(obs.station_id))
            if m is None:
                continue
            obs.lag_line = list(velocity_fit)
            obs.lag      = obs.state_vect_dist - _lineFunc(obs.time_data, *velocity_fit)
        traj_report.saveReport(
            dir_path, file_name + "_report.txt", verbose=False, save_results=True
        )
    except Exception as e:
        print("Warning: could not write curved report ({:s})".format(str(e)))
    savePickle(traj_c, dir_path, file_name + "_trajectory.pickle")
    print("Curved trajectory outputs saved to:", dir_path)


def plotCurvedVsLinear(
    traj_los,
    curved_metrics,
    output_dir=None,
    file_name="curved_vs_linear",
    show_plots=True,
    plot_file_type="png",
):
    """
    Reproduce PyLIG's diagnostic plot as linear vs curved comparisons.
    traj_los:       the Trajectory object returned by solveTrajectoryGeneric
    curved_metrics: list of dicts from computeCurvedTrajectoryMetrics
    """

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(traj_los.observations)))
    markers = [("x", 1), ("+", 1.5), ("o", 0.8), ("s", 0.8)]

    # Sort both by beginning height so colors align with PyLIG's ordering
    obs_sorted = sorted(traj_los.observations, key=lambda x: x.rbeg_ele, reverse=True)
    curved_sorted = sorted(curved_metrics, key=lambda x: x["rbeg_ele"], reverse=True)

    # ── 2. Distance from state vector vs time ────────────────────────────────
    def lineFunc(t, v, c):
        return v * t + c

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Distance from state vector vs time — linear (left) vs curved (right)")

    for ax, data_list, title in [
        (
            axes[0],
            [
                (
                    obs.time_data,
                    obs.state_vect_dist,
                    obs.model_ht,
                    obs.ignore_list,
                    str(obs.station_id),
                )
                for obs in obs_sorted
            ],
            "Linear (PyLIG)",
        ),
        (
            axes[1],
            [
                (
                    c["time_data"],
                    c["state_vect_dist"],
                    c["model_ht"],
                    c["ignore_list"],
                    str(c["station_id"]),
                )
                for c in curved_sorted
            ],
            "Curved (Keplerian arc)",
        ),
    ]:

        # collect all (time, arc_length) pairs across stations for the velocity fit
        all_t, all_svd = [], []
        for t, svd, ht, ign, sid in data_list:
            mask = ign == 0
            all_t.extend(t[mask].tolist())
            all_svd.extend(svd[mask].tolist())

        all_t = np.array(all_t)
        all_svd = np.array(all_svd)

        # fit to first 80% of points (same as PyLIG)
        sort_idx = np.argsort(all_t)
        n_fit = int(0.8 * len(all_t))
        fit_params, _ = curve_fit(
            lineFunc,
            all_t[sort_idx][:n_fit],
            all_svd[sort_idx][:n_fit],
            p0=[8000.0, 0.0],
        )
        t_line = np.linspace(all_t.min(), all_t.max(), 500)
        ax.plot(
            lineFunc(t_line, *fit_params) / 1000,
            t_line,
            "--",
            color="lightblue",
            linewidth=1.5,
            label=f"Velocity fit  v={fit_params[0]/1000:.3f} km/s",
            zorder=5,
        )

        ht_min, ht_max = np.inf, -np.inf

        for i, (t, svd, ht, ign, sid) in enumerate(data_list):
            marker = markers[i % len(markers)][0]
            ax.plot(
                svd[ign == 0] / 1000,
                t[ign == 0],
                marker=marker,
                markersize=3,
                color=colors[i],
                label=sid,
            )
            if np.any(ign):
                ax.scatter(
                    svd[ign > 0] / 1000,
                    t[ign > 0],
                    facecolors="k",
                    edgecolors=colors[i],
                    marker="o",
                    s=8,
                    zorder=4,
                )
            ht_min = min(ht_min, np.min(ht) / 1000)
            ht_max = max(ht_max, np.max(ht) / 1000)

        ax.set_title(title)
        ax.set_xlabel("Distance from state vector (km)")
        ax.invert_yaxis()
        ax.grid(True)
        ax.legend(prop={"size": 6})

        ax2 = ax.twinx()
        ax2.set_ylim(ht_min, ht_max)
        ax2.set_ylabel("Height (km)")
        ax2.invert_yaxis()

    axes[0].set_ylabel("Time (s)")
    plt.tight_layout()
    if output_dir:
        savePlot(plt, f"{file_name}_state_vect_dist.{plot_file_type}", output_dir)
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()


if __name__ == "__main__":

    import argparse
    from wmpl.Utils.Pickling import loadPickle

    arg_parser = argparse.ArgumentParser(
        description="Run the curved (Keplerian arc) trajectory solver on a saved trajectory pickle."
    )
    arg_parser.add_argument(
        "pickle_path",
        metavar="PICKLE_PATH",
        type=str,
        help="Path to the *_trajectory.pickle file produced by ECSV.py (or another WMPL solver).",
    )
    arg_parser.add_argument(
        "--extra_pickles",
        metavar="PICKLE_PATH",
        nargs="*",  # zero or more paths
        default=[],
        help="Optional trajectory pickles from additional countries/networks, in any number.",
    )
    arg_parser.add_argument(
        "--extra_ecsv_dir",
        type=str,
        default=None,
        help="Directory of ECSV files from cameras PyLIG dropped (no station overlap). ",
    )
    arg_parser.add_argument(
        "--tle_bstar", type=float, default=0.0001,
        help="The B* from the matched TLE, in TLE units (1/ER); converted to 1/m internally"
    )
    arg_parser.add_argument(
        "--hideplots",
        action="store_true",
        help="Don't show plots on screen (still saves them to disk).",
    )
    arg_parser.add_argument(
        "--imgformat",
        metavar="IMG_FORMAT",
        type=str,
        default="png",
        help="File format for saved plot images (default: png).",
    )
    cml_args = arg_parser.parse_args()

    pickle_dir = os.path.dirname(os.path.abspath(cml_args.pickle_path))
    pickle_file = os.path.basename(cml_args.pickle_path)
    traj = loadPickle(pickle_dir, pickle_file)

    all_observations = list(traj.observations)
    obs_group_sizes = [len(traj.observations)]  # group 0: primary pickle
    linear_groups = [(traj, list(traj.observations))]

    for extra_pickle_path in cml_args.extra_pickles:
        pkl_dir = os.path.dirname(os.path.abspath(extra_pickle_path))
        pkl_file = os.path.basename(extra_pickle_path)
        traj_extra = loadPickle(pkl_dir, pkl_file)
        linear_groups.append((traj_extra, list(traj_extra.observations)))
        dt_seconds = (traj_extra.jdt_ref - traj.jdt_ref) * 86400.0
        print(f"  {pkl_file}: offset {dt_seconds:.2f} s relative to primary reference.")

        shifted = []
        for obs in traj_extra.observations:
            obs_copy = copy.deepcopy(obs)
            obs_copy.time_data = obs.time_data + dt_seconds
            shifted.append(obs_copy)

        all_observations.extend(shifted)
        obs_group_sizes.append(len(shifted))  # one group per extra pickle
        print(f"  Added {len(shifted)} stations from {pkl_file}.")

    linear_observations = list(all_observations)

    if cml_args.extra_ecsv_dir:
        print("Loading extra single-station ECSVs:")
        extra_obs = loadExtraECSVsForCurvedSolver(cml_args.extra_ecsv_dir, traj.jdt_ref)
        all_observations.extend(extra_obs)
        obs_group_sizes.append(len(extra_obs))  # extra ECSVs as their own group
        print(f"  Added {len(extra_obs)} extra station(s).")

    print(
        f"Total: {len(all_observations)} stations across {len(obs_group_sizes)} groups: {obs_group_sizes}"
    )

    dir_path = traj.output_dir
    # --- diagnostic: scan speed with position and direction held at the seed ---
    _t0g = min(obs.time_data[0] for obs in all_observations)
    _t0e = jd2Date(traj.jdt_ref, dt_obj=True) + timedelta(seconds=float(_t0g))
    _f107, _f107A, _ap = get_space_weather(_t0e)
    _u = -traj.radiant_eci_mini
    _scan_args = (all_observations, traj.v_init, _u, _t0g, _t0e,
                  _f107, _f107A, _ap, obs_group_sizes,
                  60.0, 1e9, 1e9, None)     # sigma_v/sigma_dir huge => priors off

    _best = (np.inf, 0.0)
    print("speed scan (position and direction fixed at PyLIG seed, b0 = 0):")
    for dv in range(-400, 401, 50):
        p = np.concatenate([traj.state_vect_mini, _u*(traj.v_init + dv), [0.0]])
        c = angleSumOrbitalArc(p, *_scan_args)
        print(f"  v = {traj.v_init + dv:8.1f} m/s   chi2 = {c:.4e}")
        if c < _best[0]:
            _best = (c, float(dv))
    v_seed = traj.v_init + _best[1]
    print(f"  -> seeding fit at v = {v_seed:.1f} m/s (chi2 = {_best[0]:.4e})")

    # Use the PyLIG LoS result as the initial guess for the curved trajectory solver.
    # traj.observations is the list of ObservedPoints objects already populated by solveTrajectoryGeneric
    print("Fitting satellite trajectory:")
    angleSumOrbitalArc.verbose = True
    # replace n_country1_obs=n_country1, n_country2_obs=n_country2 with:
    r0_fit, v0_fit, b0_fit, t0_epoch, f107, f107A, ap_daily = (
        fitSatelliteTrajectory(
            all_observations,
            traj,
            traj.jdt_ref,
            abs(cml_args.tle_bstar)/(EARTH_RADIUS_KM*1000.0),  # TLE B* (1/ER) -> SI (1/m)
            obs_group_sizes=obs_group_sizes,
            v_init_override=v_seed
        )
    )
    angleSumOrbitalArc.verbose = False

    print("B* = ", b0_fit)
    angular_linear = {}
    sid_source = {}
    for traj_k, obs_k in linear_groups:
        res_k = computeAngularRMSDLinear(obs_k, traj_k, f"Linear (PyLIG) - {traj_k.file_name}")
        for sid in set(res_k) & set(angular_linear):
            print(f"  WARNING: station {sid} is in more than one pickle; later entry overwrites it")
        angular_linear.update(res_k)
        sid_source.update({sid: traj_k.file_name for sid in res_k})
    angular_curved = computeAngularRMSD(
        linear_observations,
        r0_fit,
        v0_fit,
        "Curved",
        b0_fit,
        t0_epoch,
        f107,
        f107A,
        ap_daily,
    )

    # The state vector epoch: jdt_ref plus the time of the first observation across all stations
    t0_global = min(obs.time_data[0] for obs in all_observations)
    state_epoch_jd = traj.jdt_ref + t0_global / 86400.0
    state_epoch_dt = jd2Date(state_epoch_jd, dt_obj=True)

    print("Computing curved trajectory metrics:")

    curved_metrics = computeCurvedTrajectoryMetrics(
        all_observations,
        r0_fit,
        v0_fit,
        b0_fit,
        t0_epoch,
        f107,
        f107A,
        ap_daily,
        traj.jdt_ref
    )

    # ── Build the side-by-side comparison table ───────────────────────────────────
    comparison_lines = [
        "Linear vs. Curved Trajectory Angular RMSD Comparison",
        "=" * 60,
        "",
        "State epoch : {:s} UTC".format(
            state_epoch_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        ),
        "Position (m): {:s}".format(str(r0_fit)),
        "Velocity m/s: {:s}".format(str(v0_fit)),
        "",
        "{:<14s}  {:>16s}  {:>16s}  {}".format(
            "Station", "Linear (arcsec)", "Curved (arcsec)", "Better?"
        ),
    ]
    for sid in sorted(angular_linear):
        lin = angular_linear.get(sid, float("nan"))
        cur = angular_curved.get(sid, float("nan"))
        flag = "yes" if cur < lin else "no"
        comparison_lines.append(
            "  {:<12s}  {:>16.3f}  {:>16.3f}  {}".format(sid, lin, cur, flag)
        )
    comparison_str = "\n".join(comparison_lines)
    print(comparison_str)

    # Write comparison to file
    comparison_path = os.path.join(dir_path, traj.file_name + "_linear_vs_curved.txt")
    with open(comparison_path, "w") as f:
        f.write(comparison_str + "\n")
    print("\nComparison table saved to:", comparison_path)

    # ── Save all curved-trajectory output files ───────────────────────────────────
    saveCurvedTrajectory(
        traj,
        r0_fit,
        v0_fit,
        curved_metrics,
        dir_path,
        all_observations=all_observations,
        b0_fit=b0_fit,
        t0_epoch=t0_epoch,
        f107=f107,
        f107A=f107A,
        ap_daily=ap_daily,
        show_plots=(not cml_args.hideplots),
        plot_file_type=cml_args.imgformat,
    )
