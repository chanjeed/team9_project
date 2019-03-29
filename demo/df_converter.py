import pandas as pd
import os
import glob

if __name__ == '__main__':

    dtype = {'id': 'object', 'x01': 'str', 'x02': 'str'}
    root_dir = r'C:\Users\Alliom\Desktop\dlb_2018\dataset'
    df = pd.read_csv(os.path.join(root_dir, 'recipes.csv'), dtype=dtype)
    for index in range(len(df)):
        df.iat[index, 0] = os.path.join(root_dir, 'recipes', os.path.basename(df.iat[index, 0]))
        df.iat[index, 1] = os.path.join(root_dir, 'recipes', os.path.basename(df.iat[index, 1]))

    df.to_csv(os.path.join(root_dir, 'recipes_.csv'), index=False)
