import os
import pandas as pd


def make_csv(x, y, filename, data_dir):
    """Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
       """
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Merge data as pandas dataframe
    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1).to_csv(
        os.path.join(data_dir, filename), index=False, header=False
    )

    # nothing is returned, but a print statement indicates that the function has run
    print("Path created: " + str(data_dir) + "/" + str(filename))
