from operator import add
import numpy as np

# Conversione di un dataset a classe singola in dataset multi classe
def to_multiclass(input1, input2, output, entity_map, relation_map):
    # Costruzione indice
    h_index = dict()
    for i in range(len(input1)):
        sbj = input1[i]
        obj = input2[i]
        out = relation_map[output[i]]
        try:
            h_index[(sbj, obj)] += np.array(out)
        except:
            h_index[(sbj, obj)] = np.array(out)
    
    #print(h_index)

    in1_matrix, in2_matrix, out_matrix = [], [], []
    for k, multi_rel in h_index.items():
        sbj, obj = k        
        in1_matrix.append(np.array(entity_map[sbj]))
        in2_matrix.append(np.array(entity_map[obj]))
        out_matrix.append(multi_rel)
    # Conversione da liste di array a matrici numpy
    in1_matrix = np.vstack(in1_matrix)
    in2_matrix = np.vstack(in2_matrix)
    out_matrix = np.vstack(out_matrix)

    return in1_matrix, in2_matrix, out_matrix




if __name__ == '__main__':

    in1 = ['pippo', 'pippo', 'pluto', 'pippo']
    in2 = ['A', 'B', 'C', 'B', ]
    out = ['x', 'y', 'z', 'j']

    e_map = {'pippo': [1,2,3], 'pluto': [4,5,6], 'A': [7,8,9], 'B': [2,2,2], 'C': [4,1,5]}
    r_map = {'x': [1, 0, 0, 0], 'y': [0, 1, 0, 0], 'z': [0, 0, 1, 0], 'j': [0, 0, 0, 1]}

    to_multiclass(in1, in2, out, e_map, r_map)
