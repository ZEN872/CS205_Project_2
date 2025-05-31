import pandas as pd

#define data set 
#Small Data Set 
df = pd.read_csv("CS205_small_Data__28.txt")
#Large Data Set 
#df = pd.read_csv("CS205_large_Data__46.txt")

print(df)

def leave_1_out_cross_vaidation ():
    return 