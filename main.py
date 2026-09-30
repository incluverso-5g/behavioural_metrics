"""
TACCESS paper - in-game behavioural metrics (Figs. 5 and 6).

Computes, from the raw VR logs, the per-session behavioural metrics of the
Cognitivo study and the log-based figures of the paper
"I will find a job": Co-Design and Longitudinal Case Study of an Immersive
Serious Game for Adults with Intellectual Disabilities (Sec. 3.7.3 / Sec. 5.3).

Data-extraction entry points (same names as in the original analysis code):

    compute_movement_stats()               -> output/movement_stats.csv
        HMD movement per session: displacement, speed, head rotations,
        time not moving, pauses (Fig. 6)
    compute_progress_metrics()             -> output/progress_stats.csv
        Events per session: successes, errors, audios, repetitions, times
    interaction_extraction_and_analysis()  -> output/interactions_summary.csv
        Object interactions per session: time / distance / number of grabs
        with correct and distractor objects (Fig. 5)

Figures:
    Fig. 5   game time + interaction time with correct / distractor objects
    Fig. 6a  number of pauses normalised by the number of successes
    Fig. 6b  average pause duration

Usage:
    python main.py                 # compute what is missing + plot both figures
    python main.py --force         # recompute everything from the raw logs
    python main.py --only fig6     # just one figure
    python main.py --show          # also open the figures in a window
    python main.py --data DIR --output DIR   # use another data / output folder

    python make_example_data.py && python main.py --data example_data --output example_output
                                   # try it on a small synthetic dataset

Sections of this file:
    1. Configuration
    2. Reading the raw logs
    3. Movement stats        compute_movement_stats()
    4. Progress stats        compute_progress_metrics()
    5. Object interactions   interaction_extraction_and_analysis()
    6. Plots                 Fig. 5 and Fig. 6
    7. Main
"""
import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

# Raw logs exported by the VR app. Layout: DATA_DIR / <user> / <session> / <file>
# (both can be changed from the command line with --data and --output)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "CognitivoAnalysisData"

# Everything the code generates goes here (the raw data is never modified)
OUTPUT_DIR = ROOT_DIR / "output"


def set_paths(data_dir=None, output_dir=None):
    """Set the data / output folders (and every path derived from them)."""
    global DATA_DIR, OUTPUT_DIR, INTERMEDIATE_DIR, FIGURES_DIR
    global MOVEMENT_STATS_CSV, PROGRESS_STATS_CSV, INTERACTIONS_SUMMARY_CSV
    if data_dir is not None:
        DATA_DIR = Path(data_dir).resolve()
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir).resolve()
    INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"   # per-session files (<user>/<session>/)
    FIGURES_DIR = OUTPUT_DIR / "figures"
    MOVEMENT_STATS_CSV = OUTPUT_DIR / "movement_stats.csv"
    PROGRESS_STATS_CSV = OUTPUT_DIR / "progress_stats.csv"
    INTERACTIONS_SUMMARY_CSV = OUTPUT_DIR / "interactions_summary.csv"


set_paths()

# Participants and sessions are detected from the folder names:
# every sub-folder of DATA_DIR is a participant, and every numeric sub-folder
# of a participant is a session (processed in numerical order).

# Pauses (Sec. 3.7.3 of the paper)
MIN_SAMPLES_PER_SECOND = 40      # 1-s bins with fewer HMD samples are dropped
PAUSE_SPEED_THRESHOLD = 0.1      # m/s -> "not walking" (Bohannon & Andrews 2011)
MIN_PAUSE_DURATION = 1           # s
# Hysteresis thresholds (m/s) used only for 'share_of_time_not_moving'
NOT_MOVING_ENTER, NOT_MOVING_EXIT = 0.03, 0.07

# --------------------------------------------------------------------------- #
# STUDY-SPECIFIC SETTINGS
# Values that only make sense for the Cognitivo dataset. With logs of the same
# VR game from another study, empty these two. With another game, also review
# the functions tagged "STUDY-SPECIFIC" below (target rules and feedback delays
# depend on the object names, the scene layout and the game logic).
# --------------------------------------------------------------------------- #
# Sessions with broken / truncated logs that cannot be recovered. They are
# skipped in the movement and progress stats. (In all three cases the level
# was repeated in a later session, which is the one used in the figures.)
EXCLUDED_SESSIONS = {
    "EC4": ["4"],
    "EC5": ["8"],
    "EC7": ["13"],
}

# Timestamps of "inactive without active" target transitions caused by level
# resets. They are expected, so no warning is printed for them.
KNOWN_RESET_TS = {
    1746444541234, 1744021640544, 1748938908247,
    1746530439850, 1748944857187, 1742895398189,
}

# Plot colours (same as in the paper)
FIG5_COLORS = {"correct": "#6C9A8B", "game_time": "#D96459", "distractor": "#2E4057"}
FIG6_SCENE_COLORS = {"caf": "blue", "sup": "orange"}


# ===========================================================================
# 2. READING THE RAW LOGS
# ===========================================================================
# Per session (DATA_DIR/<user>/<session>/):
#   Events_<u>_<s>.txt   in-game events (scene start, Lanza_Acierto, Lanza_Error, ...)
#   EyeMov_<u>_<s>.txt   HMD pose + eye tracking
#   Scene_<u>_<s>.txt    position/state of every object in the scene
#   Targets_<u>_<s>.txt  target markers written by Unity (only filled in S1)

def session_file(prefix, u, s, base=None):
    return (base or DATA_DIR) / u / s / f"{prefix}_{u}_{s}.txt"


def out_file(prefix, u, s):
    """Path of a per-session file generated by this code."""
    path = session_file(prefix, u, s, base=INTERMEDIATE_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Session metadata
# --------------------------------------------------------------------------- #
@dataclass
class SessionInfo:
    scene: str        # "caf" | "sup"
    task: int         # 0 = training, 1, 2
    level: int        # 0 = training, 1..6

    @property
    def code(self):   # "C1", "S2", "CE", ...
        return self.scene[0].upper() + ("E" if self.task == 0 else str(self.task))


_SCENE_RE = re.compile(r"Escena_(Cafeteria|Supermercado)_(?:(Entrenamiento)|Tarea_(\d)[_;]Nivel(\d))")


def get_scene_level(u, s) -> Optional[SessionInfo]:
    """Scene/task/level played in a session, read from its Events file
    (None if the session does not exist). If several scene events are
    present, the last one wins."""
    path = session_file("Events", u, s)
    if not path.is_file():
        return None
    info = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _SCENE_RE.search(line)
            if m:
                scene = "caf" if m.group(1) == "Cafeteria" else "sup"
                if m.group(2):  # Entrenamiento
                    info = SessionInfo(scene, 0, 0)
                else:
                    info = SessionInfo(scene, int(m.group(3)), int(m.group(4)))
    return info


def _natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def get_users():
    """Participant folders in DATA_DIR (natural order: P2 before P10)."""
    return sorted((d.name for d in DATA_DIR.iterdir() if d.is_dir()), key=_natural_key)


def get_sessions(u):
    """Numeric session folders of a participant, in numerical order."""
    return sorted((d.name for d in (DATA_DIR / u).iterdir() if d.is_dir() and d.name.isdigit()), key=int)


def list_sessions(include_training=True, apply_exclusions=True):
    """[(user, session, SessionInfo), ...] for every session with an Events file."""
    out = []
    for u in get_users():
        for s in get_sessions(u):
            if apply_exclusions and s in EXCLUDED_SESSIONS.get(u, []):
                continue
            info = get_scene_level(u, s)
            if info is None or (info.level == 0 and not include_training):
                continue
            out.append((u, s, info))
    return out


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def get_events(u, s):
    """Events of the actual game, cleaned:
      * starts at the first 'Escena_' event (eye-tracking calibration scene excluded),
      * ends at 'Fin_Niveles', 'Pierde3Vidas_', or the 3rd 'Nivel_finalizado_' / '3Errores_'
        (if none of these exist, it is cut at the last eye-tracking scene),
      * 'LanzaError' immediately undone by the therapist ('QuitarError') is removed.
    Timestamps are integer ms."""
    path = session_file("Events", u, s)
    if not path.is_file():
        return pd.DataFrame(columns=["timestamp", "event"])
    df = pd.read_csv(path, skiprows=2, sep=",", names=["timestamp", "event"])

    escena_mask = df["event"].str.startswith("Escena_") & (df["event"] != "Escena_EyeTrackingTest")
    start_idx = df[escena_mask].index.min()
    end_idx = len(df)

    has_end_marker = any([
        df["event"].str.contains("Fin_Niveles").any(),
        df["event"].str.contains("Pierde3Vidas_").any(),
        df["event"].str.contains("Nivel_finalizado_").sum() >= 3,
        df["event"].str.contains("3Errores_").sum() >= 3,
    ])
    if not has_end_marker:
        eye_idx = df.index[df["event"].str.contains("Escena_EyeTrackingTest")]
        if len(eye_idx) > 0 and eye_idx[-1] > 0:
            end_idx = eye_idx[-1]

    n_finished = n_3err = 0
    for i in range(start_idx, len(df)):
        ev = df.at[i, "event"]
        if ev == "Fin_Niveles" or ev.startswith("Pierde3Vidas_"):
            end_idx = i + 1
            break
        if ev.startswith("Nivel_finalizado_"):
            n_finished += 1
            if n_finished >= 3:
                end_idx = i + 1
                break
        elif ev.startswith("3Errores_"):
            n_3err += 1
            if n_3err >= 3:
                end_idx = i + 1
                break

    df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    undone = set()
    for i in range(1, len(df)):
        if df.at[i, "event"] == "QuitarError" and df.at[i - 1, "event"] == "LanzaError":
            undone.update((i - 1, i))
    return df.drop(list(undone)).reset_index(drop=True)


def count_correct_actions(events_df):
    """Number of successes ('Lanza_Acierto'), used to normalise the metrics."""
    return int((events_df["event"] == "Lanza_Acierto").sum())


def identify_repetitions(events_df):
    """Split the events into repetitions (lists of row indexes).
    A repetition starts at the scene start, after '3Errores', 'Nivel_finalizado',
    'Pierde3Vidas' or 'ResetLevel'. A 'ResetLevel' discards all previous ones."""
    repetitions, current = [], []
    for index, event in events_df["event"].items():
        if "ResetLevel" in event:
            repetitions.clear()
            current.clear()
        if any(kw in event for kw in ["Escena_", "3Errores", "Nivel_finalizado", "Pierde3Vidas", "ResetLevel"]):
            if current:
                repetitions.append(current)
            current = [index]
        else:
            current.append(index)
    if current:
        repetitions.append(current)
    # a repetition with a single event (just the start marker) is not a real one
    return [rep for rep in repetitions if len(rep) > 1]


def get_repetitions_number(u, s):
    """(events_df, repetitions, number of repetitions) of a session."""
    df = get_events(u, s)
    if len(df) == 0:
        return None, None, None
    repetitions = identify_repetitions(df)
    return df, repetitions, len(repetitions)


# --------------------------------------------------------------------------- #
# HMD movement
# --------------------------------------------------------------------------- #
_EYEMOV_COLUMNS = [
    "timestamp", "Validity", "HeadPositionX", "HeadPositionY", "HeadPositionZ",
    "HeadRotationX", "HeadRotationY", "HeadRotationZ",
    "LeftEyeOriginX", "LeftEyeOriginY", "LeftEyeOriginZ",
    "RightEyeOriginX", "RightEyeOriginY", "RightEyeOriginZ",
    "CombinedEyeOriginX", "CombinedEyeOriginY", "CombinedEyeOriginZ",
    "LeftEyeDirectionX", "LeftEyeDirectionY", "LeftEyeDirectionZ",
    "RightEyeDirectionX", "RightEyeDirectionY", "RightEyeDirectionZ",
    "CombinedEyeDirectionX", "CombinedEyeDirectionY", "CombinedEyeDirectionZ",
    "LeftEyeOpenness", "RightEyeOpenness",
    "LeftEyePupilDiameter", "RightEyePupilDiameter", "DropMe1", "DropMe2",
]


def read_eye_mov_data(u, s, events_df=None):
    """HMD samples restricted to the game time span given by the events.
    (The file header is misaligned, hence the fixed column list; the last 50
    rows are dropped because the app writes them while closing.)"""
    path = session_file("EyeMov", u, s)
    if not path.is_file():
        return None
    if events_df is None:
        events_df = get_events(u, s)
    t0, t1 = events_df["timestamp"].min(), events_df["timestamp"].max()

    df = pd.read_csv(path, sep=";", skiprows=1, decimal=",")
    df = df[:-50]
    df.columns = _EYEMOV_COLUMNS
    df = df.drop(columns=["DropMe1", "DropMe2"])
    return df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)].copy()


# --------------------------------------------------------------------------- #
# Scene objects
# --------------------------------------------------------------------------- #
def scene_log_to_df(u, s):
    """Scene_ file as a DataFrame (one row per object update, decimals with ',')."""
    with open(session_file("Scene", u, s), "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[1].strip().split(";")   # first line is metadata
    num = lambda x: float(x.replace(",", "."))
    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        p = line.strip().split(";")
        rows.append({
            "timestamp": int(p[0]), "ID": int(p[1]), "object": p[2],
            "active": p[3].lower() == "true",
            "position.x": num(p[4]), "position.y": num(p[5]), "position.z": num(p[6]),
            "rotation.x": num(p[7]), "rotation.y": num(p[8]), "rotation.z": num(p[9]),
            "scale.x": num(p[10]), "scale.y": num(p[11]), "scale.z": num(p[12]),
        })
    return pd.DataFrame(rows, columns=header)


# ===========================================================================
# 3. MOVEMENT STATS  ->  compute_movement_stats()
# ===========================================================================
# Head position is summed in 1-second windows -> speed (m/s).
# A pause is a run of windows below 0.1 m/s lasting >= 1 s (Fig. 6).

def angular_diff(a, b):
    """Smallest angle difference (degrees), accounting for wrap-around."""
    return np.abs((a - b + 180) % 360 - 180)


def find_non_movements(speed_df):
    """Not-moving flag per second with hysteresis (enter < 0.03, exit > 0.07 m/s)."""
    state = [False]

    def classify(speed):
        if speed < NOT_MOVING_ENTER:
            state[0] = True
        elif speed > NOT_MOVING_EXIT:
            state[0] = False
        return state[0]

    return speed_df["displacement"].apply(classify)


def compute_stationary_segments(speed_df):
    """Durations (s) of the pauses: runs of seconds below PAUSE_SPEED_THRESHOLD
    lasting >= MIN_PAUSE_DURATION."""
    still = (speed_df["displacement"] < PAUSE_SPEED_THRESHOLD).astype(int).reset_index(drop=True)
    t = speed_df["time_bin"].reset_index(drop=True)

    changes = still.diff().fillna(0)
    starts = changes[changes == 1].index.tolist()
    ends = changes[changes == -1].index.tolist()
    if still.iloc[0] == 1:
        starts = [0] + starts
    if still.iloc[-1] == 1:
        ends = ends + [len(still) - 1]

    durations = np.array([t.iloc[e] - t.iloc[b] for b, e in zip(starts, ends)])
    return durations[durations >= MIN_PAUSE_DURATION]


def movement_analysis(u, s):
    """Movement metrics of one session (pd.Series), or None if not available."""
    events_df, _, num_repetitions = get_repetitions_number(u, s)
    if events_df is None:
        return None
    num_aciertos = count_correct_actions(events_df)

    eye = read_eye_mov_data(u, s, events_df)
    if eye is None:
        return None
    eye["timestamp"] = eye["timestamp"] / 1000
    eye["time_bin"] = np.floor(eye["timestamp"]).astype(int)
    eye = eye.groupby("time_bin").filter(lambda x: len(x) >= MIN_SAMPLES_PER_SECOND)

    # frame-to-frame horizontal displacement (vertical ignored) and head rotations
    positions = eye[["HeadPositionX", "HeadPositionZ"]].to_numpy()
    x_rot = eye[["HeadRotationX"]].to_numpy()
    y_rot = eye[["HeadRotationY"]].to_numpy()

    df = eye.iloc[1:].copy()
    df["displacement"] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    df["rotations_x"] = angular_diff(x_rot[1:], x_rot[:-1])
    df["rotations_y"] = angular_diff(y_rot[1:], y_rot[:-1])

    speed_by_second = df.groupby("time_bin")["displacement"].sum().reset_index()
    rot_x_by_second = df.groupby("time_bin")["rotations_x"].sum().reset_index()
    rot_y_by_second = df.groupby("time_bin")["rotations_y"].sum().reset_index()

    not_moving = find_non_movements(speed_by_second)
    pauses = compute_stationary_segments(speed_by_second)
    num_pauses = len(pauses)

    total_displacement = df["displacement"].sum()
    total_rotations_x = df["rotations_x"].sum() / 360
    total_rotations_y = df["rotations_y"].sum() / 360

    return pd.Series({
        "speed_df": speed_by_second,
        "rot_x_df": rot_x_by_second,
        "rot_y_df": rot_y_by_second,
        "total_displacement": total_displacement,
        "total_displacement_per_rep": total_displacement / num_repetitions,
        "total_displacement_over_correct_choices": total_displacement / num_aciertos,
        "avg_speed": speed_by_second["displacement"].mean(),
        "total_rotations_x": total_rotations_x,
        "total_rotations_y": total_rotations_y,
        "total_rotations_x_per_rep": total_rotations_x / num_repetitions,
        "total_rotations_y_per_rep": total_rotations_y / num_repetitions,
        "total_rotations_x_over_correct_choices": total_rotations_x / num_aciertos,
        "total_rotations_y_over_correct_choices": total_rotations_y / num_aciertos,
        "share_of_time_not_moving": not_moving.mean() * 100,
        "number_of_pauses": num_pauses,
        "number_of_pauses_over_correct_choices": num_pauses / num_aciertos,   # Fig. 6a
        "pauses_per_rep": num_pauses / num_repetitions,
        "avg_pause_duration": pauses.mean() if num_pauses > 0 else 0,         # Fig. 6b
        "Num_repetitions": num_repetitions,
    })


def compute_movement_stats():
    """Movement metrics for every session (training included, broken sessions excluded).
    Returns (stats_df, full_stats): full_stats also keeps the per-second
    speed / rotation DataFrames of each session."""
    rows = []
    for u, s, info in list_sessions():
        print(f"[movement] {u} session {s} ({info.code} L{info.level})")
        stats = movement_analysis(u, s)
        if stats is not None:
            rows.append({"user": u, "session": int(s), "scene": info.scene,
                         "task": info.task, "level": info.level, **stats})
    full_stats = pd.DataFrame(rows)
    stats_df = full_stats.drop(columns=["speed_df", "rot_x_df", "rot_y_df"])
    return stats_df, full_stats


# ===========================================================================
# 4. PROGRESS STATS  ->  compute_progress_metrics()
# ===========================================================================
# Per-session performance from the events: successes, errors, audio
# instructions, number of repetitions and times.

def compute_avg_time_aciertos(rep_df):
    """Mean time (s) between a success and the previous relevant event
    (only gaps between 2 and 60 s are considered)."""
    valid_prefixes = ("Lanza_Acierto", "Lanza_Error", "Nivel_finalizado_", "3Errores", "Escena_", "ResetLevel")
    df = rep_df[rep_df["event"].str.startswith(valid_prefixes)]
    diffs = []
    for pos in range(1, len(df)):
        if df["event"].iloc[pos].startswith("Lanza_Acierto"):
            diff = (df["timestamp"].iloc[pos] - df["timestamp"].iloc[pos - 1]).total_seconds()
            if 2 < diff < 60:
                diffs.append(diff)
    return np.mean(diffs) if diffs else np.nan


def event_analysis(u, s):
    """(per-repetition DataFrame, session stats dict) of one session."""
    df = get_events(u, s)
    if len(df) == 0:
        return None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    repetition_stats = []
    for rep in identify_repetitions(df):
        rep_df = df.iloc[rep]
        repetition_stats.append({
            "repetition_start": rep_df["timestamp"].iloc[0],
            "lanza_error_count": rep_df["event"].str.contains("Lanza_Error").sum(),
            "lanza_acierto_count": rep_df["event"].str.contains("Lanza_Acierto").sum(),
            "lanza_audios_count": rep_df["event"].str.contains("LanzaAudio").sum(),
            "avg_time_between_acierto": compute_avg_time_aciertos(rep_df),
            "time_to_complete": (rep_df["timestamp"].iloc[-1] - rep_df["timestamp"].iloc[0]).total_seconds(),
        })
    repetition_df = pd.DataFrame(repetition_stats)
    return repetition_df, compute_overall_event_stats(repetition_df)


def compute_overall_event_stats(df):
    """Session-level stats from the per-repetition stats."""
    tot_error = df["lanza_error_count"].sum()
    tot_acierto = df["lanza_acierto_count"].sum()
    tot_audios = df["lanza_audios_count"].sum()
    n_reps = len(df)
    complete_reps = df[df["lanza_acierto_count"] >= 5]   # repetitions with >= 5 successes
    tot_time = df["time_to_complete"].sum()
    return {
        "Tot_error": tot_error,
        "Tot_acierto": tot_acierto,
        "Tot_audios": tot_audios,
        "Avg_error_per_rep": tot_error / n_reps,
        "Avg_acierto_per_rep": tot_acierto / n_reps,
        "Avg_audios_per_rep": tot_audios / n_reps,
        "Avg_time_acierto": df["avg_time_between_acierto"].sum() / n_reps,
        "Avg_time_to_complete_rep": (complete_reps["time_to_complete"].sum() / len(complete_reps)
                                     if len(complete_reps) else np.nan),
        "Num_repetitions": n_reps,
        "Tot_time_to_complete_level": tot_time,
        "Tot_time_divided_by_correct_choices": tot_time / tot_acierto,
    }


def compute_progress_metrics():
    """Progress metrics for every session (training included, broken sessions excluded)."""
    rows = []
    for u, s, info in list_sessions():
        print(f"[progress] {u} session {s} ({info.code} L{info.level})")
        _, stats = event_analysis(u, s)
        if stats is not None:
            rows.append({"user": u, "session": int(s), "scene": info.scene,
                         "task": info.task, "level": info.level, **stats})
    return pd.DataFrame(rows)


# ===========================================================================
# 5. OBJECT INTERACTIONS  ->  interaction_extraction_and_analysis()
# ===========================================================================
# Per session (files written to INTERMEDIATE_DIR/<u>/<s>/):
#   1. Scene_ -> ResetEvents_         moments where all objects are repositioned (S1)
#   2. Scene_ -> TargetObjects_       objects that are a valid target at each moment
#   3. Scene_ -> ObjectInteractions_  object movements = participant holding them
#   4. 2 + 3 + Targets_ + Events_ -> interactionsMetrics_
#        one row per grab with its duration/distance and whether the object
#        was a correct target or a distractor at that moment
#   5. all sessions -> interactions_summary (per-session totals, used in Fig. 5)
#   6. sanity check: successes/errors matched to grabs vs. the Events file

_CLONE_RE = r"\s*\(\d+\)|\s*\(Clone\)"


def extract_object_resets(scene_df):
    """Timestamps where > 1000 objects are updated at once (scene reset),
    keeping only resets at least 3 s apart."""
    df = scene_df[scene_df.groupby("timestamp")["timestamp"].transform("size") > 1000]
    kept, last = [], None
    for ts in df["timestamp"].sort_values().unique():
        if last is None or ts - last >= 3000:
            kept.append(ts)
            last = ts
    df = df[df["timestamp"].isin(kept)].sort_values("timestamp")
    return df.groupby("timestamp").first().reset_index()[["timestamp"]]


def extract_target_objects(scene_df, task_code):
    """Activation changes of the objects that count as targets in each task.

    STUDY-SPECIFIC: the rules below depend on the object names and the layout
    of the two virtual scenes (heights of the counter / shelves, position of
    the delivery table):
        C1  plates whose name contains "sucio" (dirty)
        C2  items with negative ID above 1.2 m (on the counter)
        S2  items with negative ID above 1.2 m and z < -2.7 (order shelf)
        S1  objects not on the table (z < -2.5); only used if the Targets_
            file written by Unity is empty
    """
    df = scene_df.sort_values(["ID", "timestamp"])
    df = df[~df["object"].str.contains("Nivel", na=False)]
    if task_code in ("C1", "CE"):
        df = df[df["object"].str.contains("sucio", na=False)]      # dirty plates
    df = df[df["active"].ne(df.groupby("ID")["active"].shift())]   # keep state changes only
    df = df.sort_values("timestamp")

    if task_code == "C2":
        df = df[(df["ID"] < 0) & (df["position.y"] > 1.2)]          # items on the counter
    elif task_code == "S2":
        df = df[(df["ID"] < 0) & (df["position.z"] < -2.7) & (df["position.y"] > 1.2)]
    elif task_code in ("S1", "SE"):
        df = df[df.groupby("timestamp")["timestamp"].transform("size") < 200]
        table_ids = df.loc[df["position.z"] < -2.5, "ID"].unique()
        df = df[~df["ID"].isin(table_ids)]

    return df.drop(columns=["rotation.x", "rotation.y", "rotation.z",
                            "scale.x", "scale.y", "scale.z"])


def extract_interactions(scene_df):
    """Rows of objects that move continuously (>3 consecutive updates),
    i.e. candidates for being held by the participant."""
    scene_df = scene_df.copy()
    scene_df["object"] = scene_df["object"].apply(
        lambda x: re.sub(_CLONE_RE, "", x) if isinstance(x, str) else x)
    full = scene_df.copy()

    # timestamps with a single update = one object moving
    df = scene_df[scene_df.groupby("timestamp")["timestamp"].transform("size") == 1].copy()
    df["delta_t"] = (df["timestamp"].shift(-1) - df["timestamp"]).fillna(0).astype("int64")

    # a gap > 150 ms breaks a movement block
    rows = []
    for _, row in df.iterrows():
        rows.append(row)
        if row["delta_t"] > 150:
            sep = row.copy()
            sep["ID"], sep["object"], sep["active"], sep["delta_t"] = 0, "separator", False, -1
            rows.append(sep)
    df = pd.DataFrame(rows).reset_index(drop=True)

    block = df["ID"].ne(df["ID"].shift()).cumsum()
    df["consecutive_movements"] = df.groupby(block)["ID"].transform("size")
    df = df[df["consecutive_movements"] > 3]

    # recover rows lost because several objects were logged at the same timestamp
    recovered = [full[(full["ID"] == obj_id) &
                      (full["timestamp"] >= g["timestamp"].min()) &
                      (full["timestamp"] <= g["timestamp"].max())]
                 for obj_id, g in df.groupby("ID")]
    if recovered:
        df = pd.concat(recovered).sort_values("timestamp").reset_index(drop=True)

    df["delta_t"] = (df["timestamp"].shift(-1) - df["timestamp"]).fillna(0).astype("int64")
    block = df["ID"].ne(df["ID"].shift()).cumsum()
    df["consecutive_movements"] = df.groupby(block)["ID"].transform("size")
    return df


def _clean_name(name):
    name = re.sub(_CLONE_RE, "", name)
    if "apple" in name:
        name = "Product_apple"
    return name.removeprefix("!!!! ")


def _read_semicolon_file(path, parse_row):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0].lstrip("﻿").strip().split(";")
    header = [h for h in header if h not in {"rotation.x", "rotation.y", "rotation.z",
                                             "scale.x", "scale.y", "scale.z"}]
    rows = [parse_row(l.strip().split(";")) for l in lines[1:] if l.strip()]
    return pd.DataFrame(rows, columns=header)


def _row_movement(p):
    return {"timestamp": int(p[0]), "ID": int(p[1]), "object": _clean_name(p[2]),
            "active": p[3].lower() == "true",
            "position.x": float(p[4]), "position.y": float(p[5]), "position.z": float(p[6]),
            "delta_t": int(p[13]), "consecutive_movements": int(p[14])}


def _row_target_object(p):
    return {"timestamp": int(p[0]), "ID": int(p[1]), "object": _clean_name(p[2]),
            "active": p[3].lower() == "true",
            "position.x": float(p[4]), "position.y": float(p[5]), "position.z": float(p[6])}


def _row_target_marker(p):
    # marker name: "<prefix>_<object name>_<ID>"
    marker = p[2]
    last_us = marker.rfind("_")
    name = marker[marker.find("_") + 1:last_us]
    if "apple" in name:
        name = "Product_apple"
    return {"timestamp": int(p[0]), "ID": int(marker[last_us + 1:]),
            "object": name.removeprefix("!!!! "), "active": p[3].lower() == "true",
            "position.x": float(p[4].replace(",", ".")),
            "position.y": float(p[5].replace(",", ".")),
            "position.z": float(p[6].replace(",", "."))}


def read_interactions_files(u, s, task_code):
    """(object movements, target objects, target markers, events) of a session."""
    movements = _read_semicolon_file(out_file("ObjectInteractions", u, s), _row_movement)
    targ_obj = _read_semicolon_file(out_file("TargetObjects", u, s), _row_target_object)
    targ_mark = _read_semicolon_file(session_file("Targets", u, s), _row_target_marker)

    # STUDY-SPECIFIC: in the cafeteria, the success/error feedback is logged with
    # a delay w.r.t. the moment the object is released (3 s in C1, 1 s in C2);
    # shift the events so that they fall inside the interaction window.
    events = get_events(u, s)
    ok, err = events["event"] == "Lanza_Acierto", events["event"] == "Lanza_Error"
    if task_code in ("C1", "CE"):
        events.loc[ok, "timestamp"] -= 3000
        events.loc[err, "timestamp"] += 500
    elif task_code == "C2":
        events.loc[ok, "timestamp"] -= 1000
        events.loc[err, "timestamp"] += 500
    return movements, targ_obj, targ_mark, events


def extract_active_targets_intervals(targets_df, verbose=False):
    """[first_ts, last_ts] intervals during which each target object is active."""
    intervals = []
    for obj_id, g in targets_df.sort_values(["ID", "timestamp"]).groupby("ID"):
        g = g.reset_index(drop=True)
        name, active, start = g.loc[0, "object"], False, None
        for ts, act in zip(g["timestamp"], g["active"]):
            if act:
                if active and verbose:
                    print(f"  WARNING {obj_id} {ts} active twice")
                if not active:
                    start, active = ts, True
            elif active:
                intervals.append({"ID": obj_id, "object": name, "first_ts": start, "last_ts": ts})
                active, start = False, None
            elif verbose and ts not in KNOWN_RESET_TS:
                print(f"  WARNING {obj_id} {ts} inactive without active")

    return (pd.DataFrame(intervals)
            .groupby(["object", "first_ts", "last_ts"], as_index=False)
            .agg({"ID": "first"})[["ID", "object", "first_ts", "last_ts"]]
            .sort_values("first_ts").reset_index(drop=True))


def analyze_interactions(movements, targ_obj, targ_mark, events):
    """One row per interaction block with duration and correct/distractor label."""
    t0, t1 = events.iloc[0]["timestamp"] + 100, events.iloc[-1]["timestamp"] + 100
    in_game = lambda d: d[(d["timestamp"] >= t0) & (d["timestamp"] <= t1)]
    movements, targ_obj, targ_mark = in_game(movements), in_game(targ_obj), in_game(targ_mark)

    # A block = consecutive updates of the same object with gaps <= 350 ms
    df = movements.copy()
    df["delta_t"] = df["timestamp"].diff().fillna(0).astype(int)
    new_block = (df["ID"] != df["ID"].shift()) | (df["delta_t"] > 350)
    df["unique_interaction_block"] = new_block.cumsum()
    df["consecutive_movements"] = df.groupby("unique_interaction_block")["unique_interaction_block"].transform("size")
    df = df[df["consecutive_movements"] > 6].copy()   # ignore tiny bumps

    d = np.sqrt(df["position.x"].diff() ** 2 + df["position.y"].diff() ** 2 + df["position.z"].diff() ** 2)
    d[df["unique_interaction_block"] != df["unique_interaction_block"].shift()] = 0
    df["step_dist"] = d

    blocks = (df.groupby("unique_interaction_block")
              .agg(first_ts=("timestamp", "first"), last_ts=("timestamp", "last"),
                   ID=("ID", "first"), object=("object", "first"),
                   total_dist=("step_dist", "sum"),
                   consecutive_movements=("consecutive_movements", "count"))
              .reset_index())
    blocks["total_time"] = blocks["last_ts"] - blocks["first_ts"]

    # Correct object = it was an active target when the grab started.
    # S1 uses the markers written by Unity; the other tasks the TargetObjects file.
    targets = extract_active_targets_intervals(targ_mark if not targ_mark.empty else targ_obj)
    targets = targets.rename(columns={"ID": "target_ID", "first_ts": "target_first_ts",
                                      "last_ts": "target_last_ts"})
    tmp = blocks.merge(targets, on="object", how="left")
    hit = (tmp["first_ts"] >= tmp["target_first_ts"]) & (tmp["first_ts"] <= tmp["target_last_ts"])
    tmp["is_correct_product"] = hit
    tmp.loc[~hit, ["target_first_ts", "target_last_ts"]] = -1
    result = (tmp.groupby(list(blocks.columns), as_index=False)
              .agg({"is_correct_product": "max", "target_first_ts": "max", "target_last_ts": "max"}))
    result["is_correct_product"] = result["is_correct_product"].astype(bool)

    # Attach the success/error feedback triggered by each interaction
    ev = events[events["event"].str.contains("Lanza_Acierto|Lanza_Error")].copy()
    ev["event_type"] = ev["event"].str.replace("Lanza_", "", regex=False)
    ev = ev.rename(columns={"timestamp": "ts"})
    result["events"] = [[] for _ in range(len(result))]
    result["events_ts"] = [[] for _ in range(len(result))]
    for i, row in result.iterrows():
        m = ev[(ev["ts"] >= row["first_ts"]) & (ev["ts"] <= row["last_ts"] + 1000)]
        if not m.empty:
            result.at[i, "events"] = m["event_type"].tolist()
            result.at[i, "events_ts"] = m["ts"].tolist()
            ev = ev.drop(m.index).reset_index(drop=True)
    return result


def run_interaction_metrics_analysis(sessions):
    """Step 4 for every session -> interactionsMetrics_<u>_<s>.txt"""
    for u, s, info in sessions:
        print(f"[interactions] metrics {u} session {s} ({info.code})")
        result = analyze_interactions(*read_interactions_files(u, s, info.code))
        result.to_csv(out_file("interactionsMetrics", u, s), sep=";", index=False)


def unify_and_aggregate_interactions(sessions, normalized=True, movement_stats=None):
    """Per-session totals of the interaction metrics (training sessions skipped).
    normalized=True divides everything by the number of successes (as in the paper).
    movement_stats is only used for 'total_dist_game' (-1 if not available)."""
    if movement_stats is None and MOVEMENT_STATS_CSV.is_file():
        movement_stats = pd.read_csv(MOVEMENT_STATS_CSV, sep=";")

    rows = []
    for u, s, info in sessions:
        if info.level == 0:
            continue
        df = pd.read_csv(out_file("interactionsMetrics", u, s), sep=";")
        events = get_events(u, s)
        n = count_correct_actions(events) if normalized else 1
        ms = 1000 * n
        correct = df[df["is_correct_product"] == True]
        distractor = df[df["is_correct_product"] == False]

        total_dist = -1
        if movement_stats is not None:
            m = movement_stats[(movement_stats["user"] == u) & (movement_stats["session"] == int(s))]
            if len(m) > 0:
                total_dist = float(m["total_displacement"].iloc[0])

        rows.append({
            "user": u, "session": int(s), "task": info.code, "level": info.level,
            "total_dist_game": total_dist / n,
            "total_dist_all": df["total_dist"].sum() / n,
            "total_dist_correct": correct["total_dist"].sum() / n,
            "total_dist_incorrect": distractor["total_dist"].sum() / n,
            "total_time_game": (events["timestamp"].iloc[-1] - events["timestamp"].iloc[0]) / ms,  # Fig. 5
            "total_time_all": df["total_time"].sum() / ms,
            "total_time_correct": correct["total_time"].sum() / ms,        # Fig. 5
            "total_time_incorrect": distractor["total_time"].sum() / ms,   # Fig. 5
            "num_interactions_per_obj": df.groupby("ID").size().mean(),
            "tot_unique_interactions": len(df) / n,
            "tot_unique_interactions_correct": len(correct) / n,
            "tot_unique_interactions_incorrect": len(distractor) / n,
        })
    return pd.DataFrame(rows).sort_values(["user", "task", "level", "session"]).reset_index(drop=True)


def interactions_results_validity(sessions):
    """Sanity check: the successes/errors matched to the grabs must equal the
    ones in the Events file, and no distractor grab may carry a success."""
    n_problems = 0
    for u, s, info in sessions:
        df = pd.read_csv(out_file("interactionsMetrics", u, s), sep=";")
        df["events"] = df["events"].apply(ast.literal_eval)
        events = get_events(u, s)
        n_ok = count_correct_actions(events)
        n_err = int((events["event"] == "Lanza_Error").sum())
        got_ok = df["events"].apply(lambda x: x.count("Acierto")).sum()
        got_err = df["events"].apply(lambda x: x.count("Error")).sum()
        wrong = df[(df["is_correct_product"] == False) & df["events"].apply(lambda x: "Acierto" in x)]

        msgs = []
        if got_ok != n_ok:
            msgs.append(f"successes: events file {n_ok} != {got_ok} matched")
        if got_err != n_err:
            msgs.append(f"errors: events file {n_err} != {got_err} matched")
        if not wrong.empty:
            msgs.append(f"{len(wrong)} distractor grab(s) with a success")
        if msgs:
            n_problems += 1
            print(f"[check] {u} session {s} ({info.code}): " + "; ".join(msgs))
    print(f"[check] {n_problems} session(s) with mismatches out of {len(sessions)}")


def interaction_extraction_and_analysis(force=False, normalized=True, movement_stats=None, check=True):
    """Full object-interaction pipeline (steps 1-6 above) for every session.
    Per-session files already generated are reused unless force=True.
    Returns the per-session summary (also saved to INTERACTIONS_SUMMARY_CSV)."""
    sessions = list_sessions(include_training=True, apply_exclusions=False)

    for u, s, info in sessions:
        if out_file("interactionsMetrics", u, s).is_file() and not force:
            continue
        print(f"[interactions] extracting {u} session {s} ({info.code})")
        scene = scene_log_to_df(u, s)
        if info.code == "S1":
            extract_object_resets(scene).to_csv(out_file("ResetEvents", u, s), sep=";", index=False)
        extract_target_objects(scene, info.code).to_csv(out_file("TargetObjects", u, s), sep=";", index=False)
        extract_interactions(scene).to_csv(out_file("ObjectInteractions", u, s), sep=";", index=False)
        run_interaction_metrics_analysis([(u, s, info)])

    summary = unify_and_aggregate_interactions(sessions, normalized=normalized, movement_stats=movement_stats)
    INTERACTIONS_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(INTERACTIONS_SUMMARY_CSV, sep=";", index=False)

    if check:
        interactions_results_validity(sessions)
    return summary


# ===========================================================================
# 6. PLOTS
# ===========================================================================
# Only the LAST attempt of each (task, level) per participant is kept, then the
# metrics are averaged across participants.

def last_attempts(df, keys):
    """One row per participant and `keys`, keeping the latest session."""
    return (df.sort_values("session")
              .drop_duplicates(subset=["user", *keys], keep="last")
              .reset_index(drop=True))


def plot_fig5(interactions_summary, save=True, show=False):
    """Fig. 5: game time and interaction time with correct / distractor objects
    (normalised by the number of successes)."""
    df = last_attempts(interactions_summary, ["task", "level"])
    mean_df = (df.groupby(["task", "level"])
                 [["total_time_game", "total_time_correct", "total_time_incorrect"]]
                 .mean().reset_index())

    c = FIG5_COLORS
    paths = []
    for letter, task in zip("abcd", ["C1", "C2", "S1", "S2"]):
        d = mean_df[mean_df["task"] == task].sort_values("level")
        if d.empty:
            continue
        plt.figure(figsize=(10, 6))
        plt.stackplot(d["level"], d["total_time_correct"], d["total_time_incorrect"],
                      labels=["Correct objects", "Distractor objects"],
                      colors=[c["correct"], c["distractor"]], alpha=0.9)
        plt.plot(d["level"], d["total_time_game"], linewidth=2, label="Game time", color=c["game_time"])

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Level", fontsize=20)
        plt.ylabel("Time (s)", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        if task == "C1":                       # legend only in the first panel
            plt.legend(loc="upper right", fontsize=18)
        plt.tight_layout()
        paths.append(_finish(f"fig5{letter}_{task}_interaction_time.png", save, show))
    return paths


def _plot_pause_metric(df, variable, ylabel, legend_loc, filename, save, show, seed):
    plt.rcParams.update({k: 14 for k in ["axes.titlesize", "axes.labelsize", "xtick.labelsize",
                                          "ytick.labelsize", "legend.fontsize", "figure.titlesize"]})
    g = sns.FacetGrid(df, col="task", hue="scene", hue_order=["caf", "sup"], sharey=True,
                      palette=FIG6_SCENE_COLORS, height=4, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="level", y=variable,
                    estimator="mean", errorbar=("ci", 95), seed=seed)
    g.axes.flat[-1].legend(title="Scene", loc=legend_loc)
    g.set_axis_labels("Level", ylabel)
    g.set_titles("Task {col_name}")
    for ax in g.axes.flat:
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    g.fig.subplots_adjust(top=0.85)
    return _finish(filename, save, show)


def plot_fig6(movement_stats, save=True, show=False, seed=0):
    """Fig. 6: pause metrics. seed fixes the bootstrap of the 95% CI bands so
    the figure is reproducible."""
    df = last_attempts(movement_stats[movement_stats["level"] > 0], ["scene", "task", "level"])
    return [
        _plot_pause_metric(df, "number_of_pauses_over_correct_choices", "# of pauses", "upper left",
                           "fig6a_normalized_number_of_pauses.png", save, show, seed),
        _plot_pause_metric(df, "avg_pause_duration", "Time (s)", "upper right",
                           "fig6b_avg_pause_duration.png", save, show, seed),
    ]


def _finish(filename, save, show):
    path = FIGURES_DIR / filename
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close("all")
    return path


# ===========================================================================
# 7. MAIN
# ===========================================================================

def cached(path, compute, force):
    """Read `path` if it exists (and not force), otherwise compute and save it."""
    if path.is_file() and not force:
        print(f"Using cached {path.name}")
        return pd.read_csv(path, sep=";")
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["fig5", "fig6"], help="plot only one figure")
    ap.add_argument("--force", action="store_true", help="recompute everything from the raw logs")
    ap.add_argument("--show", action="store_true", help="display the figures")
    ap.add_argument("--data", help=f"folder with the raw logs (default: {DATA_DIR})")
    ap.add_argument("--output", help=f"folder for the results (default: {OUTPUT_DIR})")
    args = ap.parse_args()
    set_paths(args.data, args.output)

    if not DATA_DIR.is_dir():
        raise SystemExit(f"Data folder not found: {DATA_DIR}\n"
                         "(use --data or edit DATA_DIR at the top of main.py)")
    print(f"Data:   {DATA_DIR}\nOutput: {OUTPUT_DIR}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # --- data extraction ---------------------------------------------------
    movement_stats = cached(MOVEMENT_STATS_CSV, lambda: compute_movement_stats()[0], args.force)
    cached(PROGRESS_STATS_CSV, compute_progress_metrics, args.force)  # saved, not plotted
    if INTERACTIONS_SUMMARY_CSV.is_file() and not args.force:
        print(f"Using cached {INTERACTIONS_SUMMARY_CSV.name}")
        interactions_summary = pd.read_csv(INTERACTIONS_SUMMARY_CSV, sep=";")
    else:
        interactions_summary = interaction_extraction_and_analysis(force=args.force,
                                                                   movement_stats=movement_stats)

    # --- figures -----------------------------------------------------------
    if args.only in (None, "fig5"):
        for p in plot_fig5(interactions_summary, show=args.show):
            print("Saved", p)
    if args.only in (None, "fig6"):
        for p in plot_fig6(movement_stats, show=args.show):
            print("Saved", p)


if __name__ == "__main__":
    main()
