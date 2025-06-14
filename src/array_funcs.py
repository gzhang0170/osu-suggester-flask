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
                    data_table[index][5] = 10 # Catch div by 0 just in case there are no sliders or if ratio is too large
                else:
                    data_table[index][5] = min(100, map_stats[11] / map_stats[12]) # Circle/slider ratio
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
    def find_most_similar(self, data_table, ref_index):
        # Extract reference object
        ref_map = [data_table[ref_index]]

        # Compute similarity/distance using Euclidean
        distances = euclidean_distances(data_table, ref_map).flatten()

        # Get closest indices (excluding itself)
        sorted_indices = np.argsort(distances)[1:]

        # Scale from 100 to 0, 100 is most similar using power-based falloff
        # TODO: scale is an arbitary number for similarity score, can be changed to be steeper or shallower
        scale = 1.35
        scaled_distances = np.round(100 * (1 - distances**scale), 2)
        
        return sorted_indices, scaled_distances[sorted_indices]
