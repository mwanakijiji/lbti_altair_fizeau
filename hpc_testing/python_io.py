import numpy as np
import pandas as pd

# read in test text file
df = pd.read_csv("/tmp/spalding_altair/vol_c/test_input.txt")

df["col1"] = df["col2"]

df.to_csv("/tmp/spalding_altair/vol_c/test_output.txt")

print("Done.")
