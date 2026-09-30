"""
Generate a small SYNTHETIC dataset with exactly the same file format as the
logs exported by the VR game, so that main.py can be run without the study data.

    python make_example_data.py                     # -> ./example_data
    python main.py --data example_data --output example_output

It creates 3 fake participants (P01-P03), each playing levels 1-2 of the four
tasks (C1, S1, C2, S2), i.e. 8 sessions per participant. Every session has:

    Events_<u>_<s>.txt   scene start, successes/errors, end of each repetition
    Scene_<u>_<s>.txt    spawn of the objects, objects being carried, targets
                         switched off when correctly placed
    Targets_<u>_<s>.txt  target markers (only filled in supermarket task 1)
    EyeMov_<u>_<s>.txt   HMD pose at 50 Hz (walking, carrying, pauses)

The numbers are random: the figures only show that the pipeline works.
"""
import argparse
import math
from pathlib import Path

import numpy as np

SCENES = {"C": "Cafeteria", "S": "Supermercado"}
HZ = 50                                   # HMD sampling rate
FEEDBACK_DELAY = {"C1": 2500, "C2": 500, "S1": 200, "S2": 200}   # ms after release


def fmt(x):
    """Numbers in the logs use a decimal comma."""
    return f"{x:.6f}".rstrip("0").rstrip(".").replace(".", ",") if isinstance(x, float) else str(x)


# --------------------------------------------------------------------------- #
# Objects of each task
# --------------------------------------------------------------------------- #
def make_objects(task, level, rng, next_id):
    """(correct objects, distractor objects) for one repetition.
    Each object: dict(ID, name, pos=(x, y, z), drop=(x, y, z))."""
    n_correct = 2 + level
    n_distr = level
    correct, distr = [], []

    def obj(name, pos, drop, negative_id=True):
        i = next_id()
        return {"ID": -i if negative_id else i, "name": name, "pos": pos, "drop": drop}

    rnd = lambda a, b: float(rng.uniform(a, b))
    for k in range(n_correct):
        if task == "C1":   # dirty plates on the tables -> high counter
            correct.append(obj("plato_sucio (Clone)", (rnd(-1.8, 1.8), 0.8, rnd(-2.5, 2.5)), (2.3, 1.1, 0.0)))
        elif task == "C2":  # items on the counter (y > 1.2) -> tables
            correct.append(obj(f"Food_{['burger', 'salad', 'cola', 'juice'][k % 4]}_{k}",
                               (2.2, 1.3, rnd(-1, 1)), (rnd(-1.8, 1.8), 0.8, rnd(-2.5, 2.5))))
        elif task == "S1":  # products on the floor pallet -> shelves
            correct.append(obj(f"Product_{['milk', 'water_b', 'orange_j', 'onion'][k % 4]}_{k} ({k})",
                               (rnd(-1, 1), 0.6, rnd(-2, 2)), (rnd(-2.2, -1.8), 1.2, rnd(-2, 2)), False))
        else:               # S2: products on the order shelf (z < -2.7, y > 1.2) -> boxes
            correct.append(obj(f"Product_{['milk_b', 'pencils_6', 'exotic_j', 'water_b2'][k % 4]}_{k}",
                               (rnd(-1.5, 1.5), 1.4, -2.9), (rnd(-0.5, 0.5), 0.9, 2.5)))
    for k in range(n_distr):
        if task == "C1":
            distr.append(obj("Plate_WithToastTopping (Clone)", (rnd(-1.8, 1.8), 0.8, rnd(-2.5, 2.5)), None))
        elif task == "C2":
            distr.append(obj(f"Food_pizza_{k}", (2.2, 1.3, rnd(-1, 1)), None, negative_id=False))
        elif task == "S1":
            distr.append(obj(f"Product_pepper_red_{k}", (rnd(-1, 1), 0.6, rnd(-2, 2)),
                             (rnd(-2.2, -1.8), 1.2, rnd(-2, 2)), False))
        else:
            distr.append(obj(f"Product_milk_b2_{k}", (rnd(-1.5, 1.5), 1.4, -2.9), (rnd(-0.5, 0.5), 0.9, 2.5),
                             negative_id=False))
    return correct, distr


# --------------------------------------------------------------------------- #
# One session
# --------------------------------------------------------------------------- #
def make_session(out_dir, user, sess, task, level, t_start, rng):
    scene_name = SCENES[task[0]]
    tnum = task[1]
    events, scene_rows, markers = [], [], []
    head = [(t_start - 3000, 0.0, 0.0)]          # waypoints (t, x, z) of the head
    ids = iter(range(1000 + 1000 * int(sess), 10 ** 6))
    next_id = lambda: next(ids)

    def walk_to(t, x, z, speed=0.8):
        """Walk from the last waypoint to (x, z); returns arrival time."""
        _, x0, z0 = head[-1]
        head.append((t, x0, z0))
        dist = math.hypot(x - x0, z - z0)
        t_end = t + int(1000 * max(dist / speed, 0.6))
        head.append((t_end, x, z))
        return t_end

    def stay(t, dur):
        _, x0, z0 = head[-1]
        head.append((t + dur, x0, z0))
        return t + dur

    t = t_start
    events += [(t - 2500, "Escena_EyeTrackingTest"), (t, f"Escena_{scene_name}_Tarea_{tnum};Nivel{level}"),
               (t + 800, f"LanzaAudio;Instrucciones{scene_name.lower()}tarea{tnum}")]
    t = stay(t, 6000)                            # listening to the instructions

    for rep in range(1, 4):
        correct, distr = make_objects(task, level, rng, next_id)
        spawn = t + 200
        for o in correct + distr:                # all objects appear at the same timestamp
            scene_rows.append((spawn, o["ID"], o["name"], True, *o["pos"]))
        if task == "S1":                         # Unity target markers for the missing products
            for o in correct:
                m_id = -next_id()
                markers.append((spawn, m_id, f"targetObjMarker(Clone)_{o['name'].split(' (')[0]}_{o['ID']}", True,
                                o["drop"][0], o["drop"][1], o["drop"][2]))
        t = spawn + 500

        grabs = [(o, True) for o in correct] + [(o, False) for o in distr]
        rng.shuffle(grabs)
        for o, ok in grabs:
            if rng.random() < 0.3 + 0.2 * level:     # stop to look around / plan
                t = stay(t, int(rng.uniform(1200, 3500)))
            t = walk_to(t, o["pos"][0], o["pos"][2])
            t = stay(t, int(rng.uniform(300, 900)))
            # carry the object: one Scene row every 100 ms
            gs = t
            short = (not ok) and task[0] == "C"      # cafeteria: grabbing a distractor stops the action
            drop = o["drop"] if o["drop"] is not None else o["pos"]
            dur = int(rng.uniform(900, 1300)) if short else int(rng.uniform(2200, 3800))
            n = dur // 100
            for k in range(n + 1):
                a = k / n if not short else 0.15 * k / n
                x = o["pos"][0] + a * (drop[0] - o["pos"][0])
                y = o["pos"][1] + 0.25 * math.sin(math.pi * a) + a * (drop[1] - o["pos"][1])
                z = o["pos"][2] + a * (drop[2] - o["pos"][2])
                scene_rows.append((gs + 100 * k, o["ID"], o["name"], True, x, y, z))
            ge = gs + 100 * n
            if not short:
                head.append((gs, *head[-1][1:]))
                head.append((ge, drop[0] - 0.4, drop[2]))
            else:
                t = stay(ge, 0)
            if ok:
                # placed correctly -> the target is switched off (C2/S2: the item on
                # the counter / order shelf is the one deactivated, so it keeps that position)
                off_pos = o["pos"] if task in ("C2", "S2") else drop
                scene_rows.append((ge + 400, o["ID"], o["name"], False, *off_pos))
                if task == "S1":
                    mk = next(m for m in markers if m[2].endswith(f"_{o['ID']}"))
                    markers.append((ge + 400, mk[1], mk[2], False, *mk[4:]))
                events.append((ge + FEEDBACK_DELAY[task], "Lanza_Acierto"))
            else:
                events.append((gs + 200 if short else ge + 200, "Lanza_Error"))
            t = ge + 300

        t = max(t, max(e[0] for e in events)) + 500
        events.append((t, f"Nivel_finalizado_{scene_name}_Tarea_{tnum}_{level}_Repeticion_{rep}"))
        t = stay(t, 3000)

    events.append((t, "Fin_Niveles"))
    t_end = t + 4000
    events.append((t_end, "Escena_EyeTrackingTest"))

    d = out_dir / user / str(sess)
    d.mkdir(parents=True, exist_ok=True)
    write_events(d / f"Events_{user}_{sess}.txt", user, sess, events)
    write_scene(d / f"Scene_{user}_{sess}.txt", user, sess, scene_rows, rng)
    write_targets(d / f"Targets_{user}_{sess}.txt", markers)
    write_eyemov(d / f"EyeMov_{user}_{sess}.txt", user, sess, head, t_start - 3000, t_end + 3000, rng)
    return t_end


# --------------------------------------------------------------------------- #
# Writers (same format as the files exported by the game)
# --------------------------------------------------------------------------- #
def write_events(path, user, sess, events):
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(f"UserID,{user},NSession,{sess},ManoE4,mano_izquierda\n")
        f.write("TimeStamp,EVENT\n")
        for ts, ev in sorted(events, key=lambda e: e[0]):
            f.write(f"{ts},{ev}\n")


def write_scene(path, user, sess, rows, rng):
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(f"UserID;{user};NSession;{sess}\n")
        f.write("timestamp;ID;object;active;position.x;position.y;position.z;"
                "rotation.x;rotation.y;rotation.z;scale.x;scale.y;scale.z\n")
        for ts, i, name, act, x, y, z in sorted(rows, key=lambda r: r[0]):
            rot = [float(rng.uniform(0, 360)) for _ in range(3)]
            f.write(";".join([str(ts), str(i), name, str(act), fmt(float(x)), fmt(float(y)), fmt(float(z)),
                              *map(fmt, rot), "1", "1", "1"]) + "\n")


def write_targets(path, markers):
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:   # written with a BOM by Unity
        f.write("timestamp;ID;object;active;position.x;position.y;position.z\n")
        for ts, i, name, act, x, y, z in sorted(markers, key=lambda r: r[0]):
            f.write(";".join([str(ts), str(i), name, str(act), fmt(float(x)), fmt(float(y)), fmt(float(z))]) + "\n")


def write_eyemov(path, user, sess, head, t0, t1, rng):
    ts = np.arange(t0, t1 + 50 * 1000 // HZ, 1000 // HZ)       # +50 rows (dropped by main.py)
    wt, wx, wz = (np.array(v, dtype=float) for v in zip(*sorted(head)))
    x = np.interp(ts, wt, wx) + rng.normal(0, 0.0002, len(ts))
    z = np.interp(ts, wt, wz) + rng.normal(0, 0.0002, len(ts))
    y = 1.55 + rng.normal(0, 0.002, len(ts))
    yaw = (np.degrees(np.arctan2(np.gradient(x), np.gradient(z))) + 360) % 360
    pitch = (10 + rng.normal(0, 3, len(ts))) % 360
    header = ("Timestamp;Validity;HeadPositionX;HeadPositionY;HeadPositionZ;HeadRotationX;HeadRotationY;"
              "HeadRotationZ;EyeTrackingStatus;EyeTrackingValidity;LeftEyeOriginX;LeftEyeOriginY;LeftEyeOriginZ;"
              "RightEyeOriginX;RightEyeOriginY;RightEyeOriginZ;CombinedEyeOriginX;CombinedEyeOriginY;"
              "CombinedEyeOriginZ;LeftEyeDirectionX;LeftEyeDirectionY;LeftEyeDirectionZ;RightEyeDirectionX;"
              "RightEyeDirectionY;RightEyeDirectionZ;CombinedEyeDirectionX;CombinedEyeDirectionY;"
              "CombinedEyeDirectionZ;LeftEyeOpenness;RightEyeOpenness;LeftEyePupilDiameter;RightEyePupilDiameter")
    eye = ";".join(["0"] * 22)   # eye-tracking columns: not used by main.py
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(f"UserID;{user};NSession;{sess}\n{header}\n")
        for i in range(len(ts)):
            f.write(f"{ts[i]};1111111111;{fmt(float(x[i]))};{fmt(float(y[i]))};{fmt(float(z[i]))};"
                    f"{fmt(float(pitch[i]))};{fmt(float(yaw[i]))};0;{eye}\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="example_data", help="output folder (default: example_data)")
    ap.add_argument("--users", type=int, default=3, help="number of fake participants (default: 3)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    t = 1_700_000_000_000          # fake epoch in ms
    order = [("C1", 1), ("S1", 1), ("C1", 2), ("S1", 2), ("C2", 1), ("S2", 1), ("C2", 2), ("S2", 2)]
    for u in range(1, args.users + 1):
        user = f"P{u:02d}"
        for s, (task, level) in enumerate(order, start=1):
            t = make_session(out, user, s, task, level, t + 7 * 24 * 3600 * 1000, rng)
        print(f"{user}: {len(order)} sessions")
    print(f"Synthetic dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
