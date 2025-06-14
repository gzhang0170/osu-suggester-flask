# NOTE: This only works locally. 
#       I get the monthly data dump of osu! files from data.ppy.sh and
#       run this on my local machine to generate new map tables for each month.
#       There's probably a better way to do this and run it automatically
#       but this solution works for now.
#       The process takes around an hour to finish completely.

import rosu_pp_py as rosu
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
Returns a list of rows (one per mod) with all stats,
or an empty list if parsing fails or no valid rows.
"""
def process_map_file(full_path, mods):
    try:
        bm = rosu.Beatmap(path=full_path)
    except rosu.ParseError:
        return []

    # Extract hit times
    times = []
    xs, ys = [], []
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        in_hits = False
        for line in f:
            line = line.strip()
            if not in_hits:
                if line == "[HitObjects]":
                    in_hits = True
                continue
            if not line or line.startswith("["):
                break
            parts = line.split(",")
            # time is index 2, x is 0, y is 1
            if len(parts) >= 3:
                try:
                    xs.append(int(parts[0])); ys.append(int(parts[1]))
                    times.append(int(parts[2]))
                except ValueError:
                    pass

    # Convert to numpy
    times_arr = np.array(times, dtype=np.int32)
    xs = np.array(xs, dtype=np.int32)
    ys = np.array(ys, dtype=np.int32)

    # Trim coords if mismatched
    if xs.size != ys.size:
        n = min(xs.size, ys.size)
        xs, ys = xs[:n], ys[:n]

    # Compute inter-onset stats
    diffs = np.diff(np.sort(times_arr))
    pos_diffs = diffs[diffs > 0]
    mode_ms = int(np.bincount(pos_diffs).argmax()) if pos_diffs.size >= 1 else None

    # Turning‐angle (exterior) median
    turn_angle = None
    if xs.size >= 3:
        pts = np.stack((xs, ys), axis=1)
        v1  = pts[1:-1] - pts[:-2]
        v2  = pts[2:]   - pts[1:-1]
        dots = np.einsum("ij,ij->i", v1, v2)
        n1   = np.linalg.norm(v1, axis=1)
        n2   = np.linalg.norm(v2, axis=1)
        valid = (n1>0)&(n2>0)
        if np.any(valid):
            cosθ = dots[valid]/(n1[valid]*n2[valid])
            cosθ = np.clip(cosθ, -1.0, 1.0)
            interior = np.degrees(np.arccos(cosθ))
            turn_deg = 180.0 - interior
            turn_angle = float(np.median(turn_deg))

    # Per-mod stats
    rows = []
    perf = rosu.Performance()
    perf.set_accuracy(100)
    for mod in mods:
        perf.set_mods(mod)
        perf_attrs = perf.calculate(bm)
        diff = perf.difficulty()
        diff_attrs = diff.calculate(bm)

        # Only osu!standard
        if int(diff_attrs.mode) != 0:
            continue

        # Build row
        # [beatmap_id, stars, bpm_adj, cs_adj, ar, od, hp, aim, speed, slider_factor, speed_note_count, n_circles, n_sliders, pp, mod, mode_ms, median_turn_angle]
        bm_id = int(Path(full_path).stem)

        # Manual BPM changes
        if (mod == 64 or mod == 16 + 64 or mod == 2 + 64): # DT, HRDT, EZDT
            raw_bpm = bm.bpm * 1.5
        elif (mod == 256 or mod == 16 + 256 or mod == 2 + 256): # HT, HRHT, EZHT
            raw_bpm = bm.bpm * 0.75
        else:
            raw_bpm = bm.bpm
        # Manual CS changes
        if (mod == 16 or mod == 16 + 64 or mod == 16 + 256): # HR, HRDT, HRHT
            raw_cs = min(bm.cs * 1.3, 10)
        elif (mod == 2 or mod == 2 + 64 or mod == 2 + 256): # EZ, EZDT, EZHT
            raw_cs = bm.cs * 0.5
        else:
            raw_cs = bm.cs
        # Manual ms changes
        if (mod == 64 or mod == 16 + 64 or mod == 2 + 64): # DT, HRDT, EZDT
            diff_ms = mode_ms / 1.5
        elif (mod == 256 or mod == 16 + 256 or mod == 2 + 256): # HT, HRHT, EZHT
            diff_ms = mode_ms / 0.75
        else:
            diff_ms = mode_ms

        rows.append([
            bm_id,
            diff_attrs.stars,
            raw_bpm,
            raw_cs,
            diff_attrs.ar,
            diff_attrs.od,
            diff_attrs.hp,
            diff_attrs.aim,
            diff_attrs.speed,
            diff_attrs.slider_factor,
            diff_attrs.speed_note_count,
            bm.n_circles,
            bm.n_sliders,
            perf_attrs.pp,
            mod,
            diff_ms,
            turn_angle
        ])
    return rows

"""
Builds the map stats table using the database of .osu files.
"""
def get_num_map_stats(mods, max_limit=1000):
    folder = "osu_files"
    all_files = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.endswith(".osu")
    ]

    MAX_ROWS = max_limit * len(mods)
    results = []
    with ProcessPoolExecutor() as executor:
        # Submit one task per file
        futures = { executor.submit(process_map_file, path, mods): path
                    for path in all_files }
        
        # Wrap as_completed in tqdm for progress bar in console
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Processing beatmaps"):
            try:
                file_rows = fut.result()
                if file_rows:
                    # Extend and stop at cap
                    needed = MAX_ROWS - len(results)
                    results.extend(file_rows[:needed])
                    if len(results) >= MAX_ROWS:
                        break
            except Exception:
                # Exception log
                pass

    print(f"Done: collected {len(results)} rows (cap={MAX_ROWS})")
    return results

def main():
    mods = [0, 64, 16, 256, 2, 16 + 64, 2 + 64, 16 + 256, 2 + 256] # NM, DT, HR, HT, EZ, DTHR, DTEZ, HTHR, HTEZ
    map_table = np.array(get_num_map_stats(mods, max_limit=20000000)) 

    np.save("tables/map_table.npy", map_table)

if __name__ == "__main__":
    main()