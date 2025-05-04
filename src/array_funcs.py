import rosu_pp_py as rosu
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import os

class ArrayFuncs:
    """
    Loads a numpy file into an array.
    """
    def load_numpy_array(self, filename):
        try:
            # Check if file exists
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Error: The file '{filename}' does not exist.")

            # Load the array
            array = np.load(filename, allow_pickle=True)  # Pickle: prevents loading arbitrary objects

            # print(f"Successfully loaded '{filename}'. Shape: {array.shape}, Dtype: {array.dtype}")
            return array
        except FileNotFoundError as e:
            print(e)
        except OSError as e:
            print(f"Error: The file '{filename}' is corrupted or invalid.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return None  # Return None if any error occurs

    """
    Builds the map stats table using the database of .osu files.
    NOTE: This function only works locally.
    """
    def get_num_map_stats(self, mods, max_limit=1000):
        index = 0
        failed = 0

        folder_path = "osu_files"
        # mods = [64] # DT
        #mods = [16, 256, 2, 16 + 64, 2 + 64, 16 + 256, 2 + 256] # HR, HT, EZ, DTHR, DTEZ, HTHR, HTEZ

        MAX_TABLE_SIZE = max_limit * len(mods)
        NUM_ATTRIBUTES = 15
        
        map_table = [[0 for i in range(NUM_ATTRIBUTES)] for j in range(MAX_TABLE_SIZE)]
        perf = rosu.Performance()
        perf.set_accuracy(100)

        for file_name in os.listdir(folder_path):
            total = index + failed
            print(f"Progress: {total} / {MAX_TABLE_SIZE}, {(total / MAX_TABLE_SIZE * 100):2f}%            ", end="\r")
            if total >= MAX_TABLE_SIZE:
                print(f"Success: {index}, Failed: {failed}, {(index / MAX_TABLE_SIZE * 100):2f}%")
                break

            beatmap_id = int(file_name.replace(".osu", ""))
            if beatmap_id == 0:
                failed += 1
                continue
            
            file_path = os.path.join(folder_path, file_name)
            try:
                map = rosu.Beatmap(path=file_path)
            except rosu.ParseError:
                print(f'{file_path} not found          ')
                failed += 1
                continue

            for mod in mods:
                perf.set_mods(mod)
                perf_attrs = perf.calculate(map)

                diff = perf.difficulty()
                diff_attrs = diff.calculate(map)

                if int(diff_attrs.mode) != 0:
                    failed += 1
                    continue

                map_table[index][0] = beatmap_id
                map_table[index][1] = diff_attrs.stars
                # Manual BPM changes
                if (mod == 64 or mod == 16 + 64 or mod == 2 + 64): # DT, HRDT, EZDT
                    map_table[index][2] = map.bpm * 1.5
                elif (mod == 256 or mod == 16 + 256 or mod == 2 + 256): # HT, HRHT, EZHT
                    map_table[index][2] = map.bpm * 0.75
                else:
                    map_table[index][2] = map.bpm
                # Manual CS changes
                if (mod == 16 or mod == 16 + 64 or mod == 16 + 256): # HR, HRDT, HRHT
                    map_table[index][3] = min(map.cs * 1.3, 10)
                elif (mod == 2 or mod == 2 + 64 or mod == 2 + 256): # EZ, EZDT, EZHT
                    map_table[index][3] = map.cs * 0.5
                else:
                    map_table[index][3] = map.cs
                map_table[index][4] = diff_attrs.ar
                map_table[index][5] = diff_attrs.od
                map_table[index][6] = diff_attrs.hp
                map_table[index][7] = diff_attrs.aim
                map_table[index][8] = diff_attrs.speed
                map_table[index][9] = diff_attrs.slider_factor
                map_table[index][10] = diff_attrs.speed_note_count
                map_table[index][11] = map.n_circles
                map_table[index][12] = map.n_sliders
                map_table[index][13] = perf_attrs.pp
                map_table[index][14] = mod

                index += 1

        return map_table[:index] # Truncate extra list entries

    """
    Builds the data stats table using the map stats table.
    """
    def get_data_stats(self, map_table):
        DATA_ATTRIBUTES = 8
        index = 0

        data_table = [[0 for i in range(DATA_ATTRIBUTES)] for j in range(len(map_table))]

        for map_stats in map_table:
            if map_stats[0] != 0:
                data_table[index][0] = map_stats[1] # SR
                data_table[index][1] = map_stats[2] # BPM
                data_table[index][2] = map_stats[3] # CS
                data_table[index][3] = map_stats[4] # AR
                data_table[index][4] = map_stats[9] # Slider factor
                if map_stats[12] == 0:
                    data_table[index][5] = 100 # Catch div by 0 just in case there are no sliders
                else:
                    data_table[index][5] = map_stats[11] / map_stats[12] # Circle/slider ratio
                data_table[index][6] = map_stats[7] / map_stats[8] # Aim/speed ratio
                data_table[index][7] = map_stats[10] / (map_stats[11] + map_stats[12])  # Speed/objects ratio

            index += 1

        return data_table

    """
    Scales data using standardization and the given weights.
    """
    def preprocess_data(self, objects, weights):
        # Normalize the data
        scaler = StandardScaler()
        scaled_objects = scaler.fit_transform(objects) 

        return scaled_objects * weights

    """
    Finds the top N most similar maps using Euclidean distance.
    """
    def find_most_similar(self, data_table, ref_index, top_n=100000):
        # Extract reference object
        ref_map = [data_table[ref_index]]

        # Compute similarity/distance using Euclidean
        distances = euclidean_distances(data_table, ref_map).flatten()

        # Get closest indices (excluding itself)
        sorted_indices = np.argsort(distances)[1:top_n+1]

        # Scale from 100 to 0, 100 is most similar using power-based falloff
        # TODO: scale is an arbitary number for similarity score, can be changed to be steeper or shallower
        scale = 1.5
        scaled_distances = np.round(100 * (1 - distances**scale), 2)
        
        return sorted_indices, scaled_distances[sorted_indices]
