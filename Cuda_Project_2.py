import pandas as pd
import random
import numpy as np
import torch 

#Small Data Set 
#df = pd.read_csv("CS205_small_Data__28.txt", delim_whitespace=True)

#Large Data Set 
#df = pd.read_csv("CS205_large_Data__46.txt", delim_whitespace=True)

#UCI - Data Set
#"""
#df = pd.read_csv("rocket_league_skillshots.data", delim_whitespace=True)

with open("rocket_league_skillshots.data", "r") as f:
    lines = f.readlines()[1:]

data = []
labels = []
current_label = None

for line in lines:
    values = line.strip().split()

    if len(values) == 1:
        # Label marker line
        current_label = int(values[0])
    else:
        # Regular data line
        float_values = list(map(float, values))
        data.append(float_values)
        labels.append(current_label)

# Create DataFrame and add classification column
df = pd.DataFrame(data)
df['label'] = labels
df = pd.concat([df.iloc[:, :-11], df.iloc[:, -1]], axis=1)# removes unwanted elements 
df= df[df.columns[::-1]] # filps the elements so that the classicifacion is in the first column 
#"""



def leave_1_out_cross_vaidation(Data, Current_set, k):
    data_c = Data.copy() # loacl Copy of the data 
    # The next 4 lines set the values of columns that in use to zero and then remove them. 
    keep_columns = [0] + [data_c.columns.get_loc(col) for col in Current_set + [k]]
    columns_to_zero = [i for i in range(data_c.shape[1]) if i not in keep_columns]
    data_c.iloc[:, columns_to_zero] = 0
    data_c = data_c.loc[:, (data_c != 0).any(axis=0)]

    # Convert to PyTorch and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.tensor(data_c.values, dtype=torch.float32).to(device)

    number_correctly_classified = 0

    for i in range(len(data_tensor)):
        object_to_classify = data_tensor[i, 1:]  # Exclude label
        label = data_tensor[i, 0]

        # Mask to exclude current object
        mask = torch.ones(len(data_tensor), dtype=torch.bool, device=device)
        mask[i] = False

        others = data_tensor[mask]
        labels = others[:, 0]
        features = others[:, 1:]

        # Compute L2 distances
        dists = torch.norm(features - object_to_classify, dim=1)
        min_idx = torch.argmin(dists)
        nearest_label = labels[min_idx]

        if label == nearest_label:
            number_correctly_classified += 1

    accuracy = number_correctly_classified / len(data_tensor)
    #print('Accuracy: ', accuracy)
    return accuracy

##########################################
def Forward_search(data): 
    number = 0
    results =[]
    Currnet_Elements_section = [] #empty set of values 
    for col in data.columns[1:]:
        number = number + 1
        feature_to_add = 0
        best_accuracy = 0

        print("on level ", number ,"th level of search tree")
        
        for k in data.columns[1:]:
            
            if k not in Currnet_Elements_section:
                print("--consider adding feature", k) #adding feature 
                accuracy = leave_1_out_cross_vaidation(data,Currnet_Elements_section, k)
                
                if accuracy > best_accuracy : # gets the best accuray 
                    best_accuracy = accuracy
                    feature_to_add = k

        Currnet_Elements_section.append(feature_to_add)
        print(Currnet_Elements_section)
        print('On level ', number, "'th adding feture", feature_to_add, "to current set ")
        results.append([Currnet_Elements_section.copy(), best_accuracy])  
        print("--------------------------------------------------------")

    for entry in results: #print the find results of the search 
        features, accuracy = entry
        print("{:<40} {:.4f}".format(str(features), accuracy))


##########################################
def Backward_search(data): 
    number = 0
    results =[]
    Currnet_Elements_section = data.columns[1:] #Start by including all elements. 
    for col in data.columns[1:]:
        number = number + 1
        feature_to_remove = 0
        best_accuracy = 0

        print("on level ", number ,"th level of search tree")
        
        for k in data.columns[1:]:
            temp_set = Currnet_Elements_section.copy()
            if k in Currnet_Elements_section:
                print("--consider removing feature", k)
                #Remove element k form the data set 
                temp_set = [x for x in Currnet_Elements_section if x != k]

                if(col != data.columns[1] and len(temp_set) > 1):
                    top_value = temp_set[1] 
                    temp_set = [x for x in temp_set if x != top_value]
                    accuracy = leave_1_out_cross_vaidation(data,temp_set, top_value)
                else:#The case if the data set only has 1 element left with in it
                    accuracy = leave_1_out_cross_vaidation(data,temp_set, k) 
              
                if accuracy > best_accuracy : # gets the best accuray 
                    best_accuracy = accuracy
                    feature_to_remove = k
            
        print(Currnet_Elements_section)
        print('On level ', number, "'th Removing feture", feature_to_remove, "to current set ")
        results.append([Currnet_Elements_section.copy(), best_accuracy])  
        Currnet_Elements_section = [x for x in Currnet_Elements_section if x != feature_to_remove]
        print("--------------------------------------------------------")

    for entry in results:  #print the find results of the search 
        features, accuracy = entry
        print("{:<40} {:.4f}".format(str(features), accuracy))

#####################################################################


#main funtion that runs the searches 
Forward_search(df)
Backward_search(df)
print()