import numpy as np
import os
from src.array_funcs import ArrayFuncs
from ossapi import Ossapi

MODS = {
    1: 'NF',
    2: 'EZ',
    4: 'TD',
    8: 'HD',
    16: 'HR',
    32: 'SD',
    64: 'DT',
    128: 'RX',
    256: 'HT',
}

"""
Parses mods based on the mod number.
"""
def parse_mods(mod_value):
    mod_value = int(mod_value) # Cast to avoid type errors
    return [name for bit, name in MODS.items() if mod_value & bit]

# TODO: Add these to the data table directly in the build
# TODO: Possibly find a better fit for all these attributes' scaling
"""
Scalers for various attributes.
"""
def exp_scale(x, exp=2):
    return np.power(exp, x)

def log_scale(x, base=2):
    return np.emath.logn(base, x + 1)

def logistic_scale(x, L=1.0, k=1.0, x0=0.0):
    return L / (1 + np.exp(-k * (x - x0)))

"""
Takes a beatmap id as input and returns an array of the most similar maps.
The map must have a leaderboard (ranked, loved, approved)
"""
def get_similar_maps(beatmap_id, mods=0, exclude_mods=[], max_maps=10):
    af = ArrayFuncs()

    # Get cached tables and load them into tables
    current_directory = os.getcwd()
    data_table_filename = os.path.join(current_directory, "src", "tables", "data_table.npy")
    data_table = af.load_numpy_array(data_table_filename)

    # Beatmap not found, return None
    if data_table is None:
        return None

    # Get index of the current map in the table
    ref_index = np.where((data_table[:, 8] == beatmap_id) & (data_table[:, 9] == mods))[0]
    if (len(ref_index) == 0):
        return None
    
    ref_index = ref_index[0]
    bpm = data_table[ref_index][1]

    # TODO: Temporary solution to solve outliers, remove for next table build
    data_table[:, 0] = np.minimum(data_table[:, 0], 12)
    data_table[:, 1] = np.minimum(data_table[:, 1], 800)
    data_table[:, 5] = np.minimum(data_table[:, 5], 10)
    data_table[:, 6] = np.minimum(data_table[:, 6], 2)

    # Scale avarious attributes
    # TODO: Add these to the data table directly in the build
    data_table[:, 2] = log_scale(data_table[:, 2], base=1.3)
    data_table[:, 3] = exp_scale(data_table[:, 3], exp=1.2)
    data_table[:, 4] = exp_scale(data_table[:, 4], exp=10)
    data_table[:, 5] = log_scale(data_table[:, 5], base=1.2)
    data_table[:, 6] = logistic_scale(data_table[:, 6], L=1, k=8, x0=1.2)
    data_table[:, 7] = logistic_scale(data_table[:, 7], L=1, k=10, x0=0.3)

    # Determine a "standardized" BPM compared to the current map
    # NOTE: This is only based on halved/doubled BPMS or similar.
    #       Not sure of a better way to implement this, especially for maps with different time signatures (ex. 1/3)
    orig_bpms = data_table[:, 1]
    factors = np.array([0.25, 0.5, 1.0, 2.0, 4.0])

    stdized = np.empty_like(orig_bpms)
    min_diff = np.full(orig_bpms.shape, np.inf)

    for f in factors:
        scaled = orig_bpms * f 
        diff = np.abs(scaled - bpm) 
        mask = diff < min_diff
        stdized[mask] = scaled[mask]
        min_diff[mask] = diff[mask]

    data_table[:, 1] = stdized

    # SR, BPM, CS, AR, Slider factor, Circle/slider ratio, Aim/speed ratio, Speed/objects ratio
    # NOTE: This is based on my personal testing of what "finds" similar maps currently based on these stats.
    #       Will likely change with playtesting and more feedback.
    weights = [1.2, 1.4, 0.6, 1.1, 0.4, 1, 2.2, 0.8]
    weights = np.multiply(weights, 0.7)

    # Standardize the table's statistics with the weights
    stdized_table = af.preprocess_data(data_table[:, :8], weights)

    # Find the indices in the map table of the most similar maps
    similar_indices, distances = af.find_most_similar(stdized_table, ref_index)
    
    # Build a dictionary of the most similar beatmap ids
    beatmaps = {}
    max_maps_i = 0
    i = 0
    for index in similar_indices:
        if data_table[index][9] in exclude_mods:
            i += 1
            continue
        id = data_table[index][8]
        mods = data_table[index][9]
        beatmaps[(id, mods)] =  {   
            "difficulty_rating": data_table[index][0],
            "bpm": data_table[index][1],
            "cs": data_table[index][2],
            "drain": data_table[index][11],
            "accuracy": data_table[index][10],
            "ar": data_table[index][3],
            "distance": distances[i]
        }
        i += 1
        max_maps_i += 1
        if max_maps_i >= max_maps:
            break
    
    return beatmaps


"""
Makes a call to the osu!api to get beatmap information.
Builds the json object with the necessary data for the frontend.
"""
def build_json(beatmaps):
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    api = Ossapi(client_id, client_secret)

    beatmap_ids = [beatmap[0] for beatmap in beatmaps]
    beatmaps_info = api.beatmaps(beatmap_ids)

    attributes = {}
    for bm in beatmaps_info:
        attributes[bm.id] = {
            "url":              bm.url,
            "card":             bm._beatmapset.covers.card,
            "artist":           bm._beatmapset.artist,
            "title":            bm._beatmapset.title,
            "version":          bm.version,
            "creator":          bm._beatmapset.creator,
            "total_length":     bm.total_length,
            "hit_length":       bm.hit_length,
            "playcount":        bm.playcount,
            "status":           bm.status.name,
            "ranked_date":      bm._beatmapset.ranked_date,
        }

    beatmap_info = []
    for (id, mods), attrs in beatmaps.tems():
        mods_list = parse_mods(mods)
        if "DT" in mods_list:
            length_mult = 0.667
        elif "HT" in mods_list:
            length_mult = 1.333
        else:
            length_mult = 1
        
        total_length = round(attributes[id]["total_length"] * length_mult)
        hit_length = round(attributes[id]["hit_length"] * length_mult)
        mods_string = ','.join(mods_list)

        attributes.append({
            "id":               id,
            "url":              attributes[id]["url"],
            "card":             attributes[id]["card"],
            "artist":           attributes[id]["artist"],
            "title":            attributes[id]["title"],
            "version":          attributes[id]["version"],
            "creator":          attributes[id]["creator"],
            "mods":             mods_string,
            "difficulty_rating":np.float64(attrs["difficulty_rating"]),
            "total_length":     total_length,
            "hit_length":       hit_length,
            "cs":               np.float64(attrs["cs"]),
            "drain":            np.float64(attrs["drain"]),
            "accuracy":         np.float64(attrs["accuracy"]),
            "ar":               np.float64(attrs["ar"]),
            "bpm":              np.float64(attrs["bpm"]),
            "playcount":        attributes[id]["playcount"],
            "status":           attributes[id]["status"],
            "ranked_date":      attributes[id]["ranked_date"],
            "distance":         np.float64(attrs["distance"])
        })
    
    sorted_beatmap_info = sorted(beatmap_info, key=lambda x: x["distance"], reverse=True)
    return sorted_beatmap_info

