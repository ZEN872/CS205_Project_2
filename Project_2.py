import pandas as pd
import random

#define data set 
#Small Data Set 
df = pd.read_csv("CS205_small_Data__28.txt", delim_whitespace=True)

#Large Data Set 
#df = pd.read_csv("CS205_large_Data__46.txt")


#Basic Test:
#print("Columns:", df.columns)
#print("Number of columns:", len(df.columns))
#print(df)

def leave_1_out_cross_vaidation(data, Current_set, k):
    return random.randint(1, 10)

def Featsure_search(data): 
    number = 0
    Currnet_Elements_section = []
    for col in data.columns[1:]:
        number = number + 1
        feature_to_add = 0
        best_accuracy = 0
        print("on level ", number ,"th level of search tree")
        for k in data.columns[1:]:
            if k not in Currnet_Elements_section:
                print("--consider adding feature", k)
                accuracy = random.randint(1, 10)#leave_1_out_cross_vaidation(data,Currnet_Elements_section, k+1)
                if accuracy > best_accuracy : 
                    best_accuracy = accuracy
                    feature_to_add = k
        Currnet_Elements_section.append(feature_to_add)
        #print(Currnet_Elements_section)
        print('On level ', number, "'th adding feture", feature_to_add, "to current set ")
        print("--------------------------------------------------------")



Featsure_search(df)