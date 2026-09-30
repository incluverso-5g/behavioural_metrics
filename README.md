# TACCESS paper – in-game behavioural metrics

Code used to extract the behavioural metrics from the game logs and to produce
Figs. 5 and 6 of *"I will find a job": Co-Design and Longitudinal Case Study of an
Immersive Serious Game for Adults with Intellectual Disabilities* (Sec. 3.7.3 and 5.3).

The study data are not distributed (they belong to the participants of the study).
A generator of **synthetic data in the same format** is included so the code can be
run and inspected, and applied to other datasets recorded with the same VR game.

## Quick start (synthetic data)

```bash
pip install -r requirements.txt          # Python >= 3.9
python make_example_data.py              # -> example_data/  (3 fake participants, 24 sessions)
python main.py --data example_data --output example_output
```

This produces the three metric tables and the two figures in `example_output/`.
The values are random; they only show that the pipeline works end to end.

## Usage

```bash
python main.py --data PATH/TO/DATA        # compute what is missing + plot the figures
python main.py --data ... --force         # recompute everything from the raw logs
python main.py --data ... --only fig6     # plot just one figure
python main.py --data ... --output DIR    # where results go (default: ./output)
```

Without `--data`, `main.py` looks for `../CognitivoAnalysisData` (the layout used in the study).

## Outputs

| Function | File | Content (one row per session) |
|---|---|---|
| `compute_movement_stats()` | `movement_stats.csv` | displacement, speed, head rotations, time not moving, pauses |
| `compute_progress_metrics()` | `progress_stats.csv` | successes, errors, audio instructions, repetitions, times |
| `interaction_extraction_and_analysis()` | `interactions_summary.csv` | time / distance / number of grabs with correct and distractor objects |
| `plot_fig5()` | `figures/fig5*.png` | game time and interaction time with correct / distractor objects |
| `plot_fig6()` | `figures/fig6*.png` | normalised number of pauses and average pause duration |

`interaction_extraction_and_analysis()` also writes per-session files to
`intermediate/<participant>/<session>/` (`ResetEvents_`, `TargetObjects_`,
`ObjectInteractions_`, `interactionsMetrics_`) and prints a sanity check: the
successes/errors assigned to each grab must match those in the Events file.

The functions can also be used from another script:

```python
import main
main.set_paths(data_dir="my_data", output_dir="my_output")
movement_stats, _ = main.compute_movement_stats()
progress_stats = main.compute_progress_metrics()
summary = main.interaction_extraction_and_analysis()
```

## Input data format

One folder per participant and one numeric sub-folder per session. Participants and
sessions are detected from the folder names (sessions are processed in numerical order,
which matters because only the last attempt of each level is kept in the figures).

```
DATA/
├── P01/
│   ├── 1/
│   │   ├── Events_P01_1.txt
│   │   ├── EyeMov_P01_1.txt
│   │   ├── Scene_P01_1.txt
│   │   └── Targets_P01_1.txt
│   ├── 2/ ...
└── P02/ ...
```

All timestamps are Unix time in **milliseconds**. Decimal numbers use a **comma**
(`1,184854`). Other files in the folders are ignored.

### `Events_<p>_<s>.txt` – in-game events (comma-separated)

```
UserID,P01,NSession,1,ManoE4,mano_izquierda        <- metadata line (ignored)
TimeStamp,EVENT                                    <- header
1700604797500,Escena_EyeTrackingTest
1700604800000,Escena_Cafeteria_Tarea_1;Nivel1
1700604815751,Lanza_Acierto
...
```

Events used by the code:

| Event | Meaning |
|---|---|
| `Escena_<Cafeteria\|Supermercado>_Tarea_<1\|2>;Nivel<1-6>` | start of a level (defines scene, task and level of the session) |
| `Escena_<Cafeteria\|Supermercado>_Entrenamiento` | start of a training session (level 0) |
| `Escena_EyeTrackingTest` | eye-tracker calibration scene (not part of the game) |
| `Lanza_Acierto` / `Lanza_Error` | success / error feedback |
| `Nivel_finalizado_...` | end of a repetition (a level = 3 successful repetitions) |
| `3Errores_...`, `Pierde3Vidas_...`, `ResetLevel`, `Fin_Niveles` | failed repetition, lost all lives, level restarted, end of the game |
| `LanzaError` followed by `QuitarError` | error launched and undone by the therapist (both removed) |
| `LanzaAudio;...` | audio instruction |

The game period of a session goes from the first `Escena_` event (not the eye-tracking
one) to `Fin_Niveles`, `Pierde3Vidas_` or the 3rd `Nivel_finalizado_` / `3Errores_`.

### `EyeMov_<p>_<s>.txt` – HMD pose (semicolon-separated, ≥ 40 Hz)

Line 1: metadata. Line 2: header. Then one row per sample with 30 fields:
`timestamp; validity; HeadPositionX; HeadPositionY; HeadPositionZ; HeadRotationX; HeadRotationY; HeadRotationZ;` followed by 22 eye-tracking fields.
Only the timestamp and the head position/rotation are used (positions in metres,
rotations in degrees). The last 50 rows are discarded (written while the app closes).

### `Scene_<p>_<s>.txt` – state of the scene objects (semicolon-separated)

```
UserID;P01;NSession;1
timestamp;ID;object;active;position.x;position.y;position.z;rotation.x;rotation.y;rotation.z;scale.x;scale.y;scale.z
1700604800200;-3000;plato_sucio (Clone);True;-1,2;0,8;0,53;...
```

One row each time an object is created, moved or (de)activated. An object that is
moved alone with updates every ≤ 150 ms is considered to be held by the participant;
a correctly placed target object gets a row with `active = False`.

### `Targets_<p>_<s>.txt` – target markers (supermarket task 1 only)

```
timestamp;ID;object;active;position.x;position.y;position.z     <- header (no metadata line)
1701209714411;-3004;targetObjMarker(Clone)_Product_milk_0_3000;True;-2,17;1,2;0,22
```

`object` is `<prefix>_<product name>_<product ID>`. `active` is `True` while the
product is missing from the shelf. In the other tasks the file contains only the header.

## Study-specific settings

These depend on the Cognitivo study or on the design of the game. With data from
another study using the same game, empty the first two. With a different game, the
last two must be adapted:

| Where in `main.py` | What |
|---|---|
| `EXCLUDED_SESSIONS` | sessions with broken logs in the original study |
| `KNOWN_RESET_TS` | timestamps whose warnings are silenced (original study) |
| `extract_target_objects()` | which objects are targets in each task (names, heights, positions of counter/shelves) |
| `read_interactions_files()` | delay of the success/error feedback in the cafeteria tasks |

## Method summary

- **Normalisation**: all metrics except the average pause duration are divided by the
  number of successes (`Lanza_Acierto`) of the level.
- **Last attempt**: in the figures, when a participant repeated a level, only the last
  session is kept; training sessions are not plotted (they are kept in the CSV files).
- **Pauses**: horizontal head displacement summed in 1-s windows; a pause is a run of
  windows below 0.1 m/s lasting ≥ 1 s.
- **Interactions**: blocks of consecutive updates of the same object (gaps ≤ 350 ms,
  > 6 updates) are grabs. A grab is on a *correct* object if the object was an active
  target when the grab started, otherwise on a *distractor*.
- Error bands in Fig. 6 are seaborn's bootstrapped 95 % CI (seed fixed).

## Structure of `main.py`

```
1. Configuration          paths, thresholds, study-specific settings, colours
2. Reading the raw logs   Events / EyeMov / Scene readers, repetitions, session metadata
3. Movement stats         compute_movement_stats()               (Fig. 6)
4. Progress stats         compute_progress_metrics()
5. Object interactions    interaction_extraction_and_analysis()  (Fig. 5)
6. Plots                  Fig. 5 and Fig. 6
7. Main                   command-line entry point
```

`make_example_data.py` generates the synthetic dataset.
