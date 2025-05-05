# NOTE: This only works locally. 
#       I get the monthly data dump of osu! files from data.ppy.sh and
#       run this on my local machine to generate new map tables for each month.
#       There's probably a better way to do this and run it automatically
#       but this solution works for now.
#       The process takes around 2 hours to finish one mod, currently
#       working on building the table for DT.

# TODO: Build the full data table instead of separate map and data tables.
#       Most likely will need to split table in 2 due to 100MB GitHub file size limit.
from array_funcs import ArrayFuncs
import numpy as np

def main():
    af = ArrayFuncs()

    map_table = np.array(af.get_num_map_stats(mods=[0], max_limit=200000)) 
    data_table = np.array(af.get_data_stats(map_table))

    data_table_full = np.hstack([
        data_table,
        map_table[:, [0]],
        map_table[:, [14]],
        map_table[:, [5]],
        map_table[:, [6]]
    ])

    np.save("tables/data_table_nm.npy", data_table_full)

    # np.save("tables/map_table_25_05_01_dthr.npy", map_table)
    # np.save("tables/data_table_25_05_01_dthr.npy", data_table)

if __name__ == "__main__":
    main()