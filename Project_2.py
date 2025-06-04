import pandas as pd
import random
import numpy as np

#define data set 
#Small Data Set 
df = pd.read_csv("CS205_small_Data__28.txt", delim_whitespace=True)

#Large Data Set 
#df = pd.read_csv("CS205_large_Data__46.txt")

#Basic Test:
#print("Columns:", df.columns)
#print("Number of columns:", len(df.columns))
#print(df)

def leave_1_out_cross_vaidation(Data, Current_set, k):
    #value = random.randint(1, 10)
    data_c = Data.copy()
    keep_columns = [0] + [data_c.columns.get_loc(col) for col in Current_set + [k]]
    columns_to_zero = [i for i in range(data_c.shape[1]) if i not in keep_columns]
    data_c.iloc[:, columns_to_zero] = 0
    data_c = data_c.loc[:, (data_c != 0).any(axis=0)]
    #print(columns_to_zero)
    #print(data_c)

    number_correctly_classifed = 0 
    for i in range(len(data_c)):
        object_to_classify = data_c.iloc[i, 1:]  # All columns except the first
        label_object_to_classify = data_c.iloc[i, 0]  # First column (label)
        
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range (len(data_c)):
            if k != i:
                comparison_object = data_c.iloc[k, 1:].to_numpy()
                distance = np.sqrt(np.sum((object_to_classify - comparison_object) ** 2))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data_c.iloc[nearest_neighbor_location, 0]
        if label_object_to_classify == nearest_neighbor_label:
                number_correctly_classifed = number_correctly_classifed + 1
    #return value
    accuracy = number_correctly_classifed / len(data_c)
    return accuracy




def Featsure_search(data): 
    number = 0
    results =[]
    Currnet_Elements_section = []
    for col in data.columns[1:]:
        number = number + 1
        feature_to_add = 0
        best_accuracy = 0
        print("on level ", number ,"th level of search tree")
        for k in data.columns[1:]:
            if k not in Currnet_Elements_section:
                print("--consider adding feature", k)
                accuracy = leave_1_out_cross_vaidation(data,Currnet_Elements_section, k)
                if accuracy > best_accuracy : 
                    best_accuracy = accuracy
                    feature_to_add = k
        Currnet_Elements_section.append(feature_to_add)
        print(Currnet_Elements_section)
        print('On level ', number, "'th adding feture", feature_to_add, "to current set ")
        results.append([Currnet_Elements_section.copy(), best_accuracy])  
        print("--------------------------------------------------------")

    for entry in results:
        features, accuracy = entry
        print("{:<40} {:.4f}".format(str(features), accuracy))



#df =df[1:10]
Featsure_search(df)
print()