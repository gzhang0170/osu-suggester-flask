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

"""

Takes a beatmap id as input and returns an array of the most similar maps.
The map must have a leaderboard (ranked, loved, approved)
"""
# TODO: Add mods into similar maps algorithm
def get_similar_maps(beatmap_id, mods=0, max_maps=10):
    af = ArrayFuncs()

    # Get cached tables and load them into tables
    current_directory = os.getcwd()
    map_table_filename_nm = os.path.join(current_directory, "src", "tables", "map_table_25_05_01_nm.npy")
    data_table_filename_nm = os.path.join(current_directory, "src", "tables", "data_table_25_05_01_nm.npy")
    map_table_nm = af.load_numpy_array(map_table_filename_nm)
    data_table_nm = af.load_numpy_array(data_table_filename_nm)

    map_table_filename_dt = os.path.join(current_directory, "src", "tables", "map_table_25_05_01_dt.npy")
    data_table_filename_dt = os.path.join(current_directory, "src", "tables", "data_table_25_05_01_dt.npy")
    map_table_dt = af.load_numpy_array(map_table_filename_dt)
    data_table_dt = af.load_numpy_array(data_table_filename_dt)

    map_table_filename_hr = os.path.join(current_directory, "src", "tables", "map_table_25_05_01_hr.npy")
    data_table_filename_hr = os.path.join(current_directory, "src", "tables", "data_table_25_05_01_hr.npy")
    map_table_hr = af.load_numpy_array(map_table_filename_hr)
    data_table_hr = af.load_numpy_array(data_table_filename_hr)

    map_table = np.concatenate((map_table_nm, map_table_dt, map_table_hr), axis=0)
    data_table = np.concatenate((data_table_nm, data_table_dt, data_table_hr), axis=0)

    # Beatmap not found, return None
    if map_table is None or data_table is None:
        return None

    # Get index of the current map in the table
    ref_index = np.where((map_table[:, 0] == beatmap_id) & (map_table[:, -1] == mods))[0][0]
    bpm = data_table[ref_index][1]

    # Determine a "normalized" BPM compared to the current map
    # NOTE: This is only based on halved/doubled BPMS or similar.
    #       Not sure of a better way to implement this, especially for maps with different time signatures (ex. 1/3)
    for data_stats in data_table:
        bpms = np.array([data_stats[1] / 4, data_stats[1] / 2, data_stats[1], data_stats[1] * 2, data_stats[1] * 4])
        i = np.argmin(np.abs(bpms - bpm))
        stdized_bpm = bpms[i]
        data_stats[1] = stdized_bpm

    # SR, BPM, CS, AR, Slider factor, Circle/slider ratio, Aim/speed ratio, Speed/objects ratio
    # NOTE: This is based on my personal testing of what "finds" similar maps currently based on these stats.
    #       Will likely change with playtesting and more feedback.
    weights = [1.4, 1, 0.6, 0.8, 0.4, 1.2, 2, 0.7]

    # Standardize the table's statistics with the weights
    stdized_table = af.preprocess_data(data_table, weights)

    # Find the indices in the map table of the most similar maps
    similar_indices, distances = af.find_most_similar(stdized_table, ref_index)
    
    # Build a dictionary of the most similar beatmap ids
    beatmaps = {}
    i = 0
    for index in similar_indices:
        id = map_table[index][0]
        beatmaps[id] =  {   
            "difficulty_rating": map_table[index][1],
            "bpm": map_table[index][2],
            "cs": map_table[index][3],
            "drain": map_table[index][6],
            "accuracy": map_table[index][5],
            "ar": map_table[index][4],
            "mods": map_table[index][14],
            "distance": distances[i]
        }
        i += 1
        if i >= max_maps:
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

    beatmap_ids = list(beatmaps.keys())
    beatmaps_info = api.beatmaps(beatmap_ids)

    attributes = []
    for bm in beatmaps_info:
        mods = parse_mods(beatmaps[bm.id]["mods"])
        if "DT" in mods:
            length_mult = 1.5
        elif "HT" in mods:
            length_mult = 0.75
        else:
            length_mult = 1

        total_length = round(bm.total_length / length_mult)
        hit_length = round(bm.hit_length / length_mult)
        mods_string = ''.join(mods)

        attributes.append({
            "id":               bm.id,
            "url":              bm.url,
            "card":             bm._beatmapset.covers.card,
            "artist":           bm._beatmapset.artist,
            "title":            bm._beatmapset.title,
            "version":          bm.version,
            "creator":          bm._beatmapset.creator,
            "mods":             mods_string,
            "difficulty_rating":beatmaps[bm.id]["difficulty_rating"],
            "total_length":     total_length,
            "hit_length":       hit_length,
            "cs":               beatmaps[bm.id]["cs"],
            "drain":            beatmaps[bm.id]["drain"],
            "accuracy":         beatmaps[bm.id]["accuracy"],
            "ar":               beatmaps[bm.id]["ar"],
            "bpm":              beatmaps[bm.id]["bpm"],
            "playcount":        bm.playcount,
            "status":           bm.status.name,
            "ranked_date":      bm._beatmapset.ranked_date,
            "distance":         beatmaps[bm.id]["distance"]
        })
    
    sorted_attibutes = sorted(attributes, key=lambda x: x['distance'], reverse=True)
    return sorted_attibutes

