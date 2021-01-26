# Keras Link Predictor (KLIP) by Enrico Verdolotti
# Link Prediction Home Made with Neural Networks :)

from os.path import join
import numpy as np
import random
import keras
from keras import layers 

# Questo oggetto espone questi metodi per l'utilizzo:
# train_model: avvia un addestramento ed utilizza il modello appena addestrato
# load_model: carica un modello preesistente da disco e le strutture dati necessarie
# save_model: salva il modello corrente su disco e le strutture dati necessarie
# predict: effettua una predizione su una o più coppie di entità
class KerasLinkPredictor:
    
    # Inizializza il modello con dimensione input ed embeddings
    def __init__(self, input_dim=10, embeddings_dim=50):
        self.input_dim = input_dim
        self.embeddings_dim = embeddings_dim


    # Carica il dataset nelle strutture necessarie
    def load_dataset(self, dataset_folder):
        required_files = ['train.tsv', 'valid.tsv', 'test.tsv']
        self.relation_map = {'not_related': None}
        self.entity_map = dict()
        self.train_input1 = list()
        self.train_input2 = list()
        self.train_output = list()
        self.valid_input1 = list()
        self.valid_input2 = list()
        self.valid_output = list()
        self.test_input1 = list()
        self.test_input2 = list()
        self.test_output = list()     
        for f in required_files:
            with open(join(dataset_folder, f), 'r') as tsv_file:
                for line in tsv_file:
                    h, r, t = line.split()
                    # Imposta input ed output 
                    if f == 'train.tsv':
                        self.train_input1.append(h)
                        self.train_input2.append(t)
                        self.train_output.append(r)
                    elif f == 'valid.tsv':
                        self.valid_input1.append(h)
                        self.valid_input2.append(t)
                        self.valid_output.append(r)
                    elif f == 'test.tsv':
                        self.test_input1.append(h)
                        self.test_input2.append(t)
                        self.test_output.append(r)                    
                    # Imposta strutture per la codifica
                    if h not in self.entity_map:
                        self.entity_map[h] = np.random.rand(self.input_dim).tolist()
                    if t not in self.entity_map:
                        self.entity_map[t] = np.random.rand(self.input_dim).tolist()
                    if r not in self.relation_map:
                        self.relation_map[r] = None
        # Aggiungi codifica OneHot per ciascuna relazione
        self.output_dim = len(self.relation_map)
        for i, rel in enumerate(self.relation_map.keys()):
            self.relation_map[rel] = np.zeros(self.output_dim).tolist()
            self.relation_map[rel][i] = 1.


    # Genera sample negativi (naive: campionati a caso)
    def gen_neg_samples(self, sbj_l, rel_l, obj_l, alpha=0.5):
        size = int(len(rel_l) * alpha)
        rel_l + ['not_related' for _ in range(size)]
        sbj_l + random.choices(list(self.entity_map.keys()), k=size)
        obj_l + random.choices(list(self.entity_map.keys()), k=size)
    

    # Codifica una lista di categorie in una matrice numpy
    # associando ad ogni valore un vettore riga secondo il mapping
    def matrix_encode(self, cat_list, cat_mapping):
        return np.array([cat_mapping[x] for x in cat_list])


    # Codifica i dati in base al nome 'train', 'test', 'split'
    # utile per non dover ripetere l'istruzione più volte 
    def data_encode(self, name):
        in1 = name + '_input1'
        in1_matrix = self.matrix_encode(getattr(self, in1), self.entity_map)
        setattr(self, in1, in1_matrix)        
        in2 = name + '_input2'
        in2_matrix = self.matrix_encode(getattr(self, in2), self.entity_map)
        setattr(self, in2, in2_matrix)        
        out = name + '_output'
        out_matrix = self.matrix_encode(getattr(self, out), self.relation_map)
        setattr(self, out, out_matrix)


    # Costruttore del modello    
    def build_model(self):        
        # Rete neurale per il soggetto
        input_sbj = keras.Input(shape=(self.input_dim,))
        hidden_sbj = layers.Dense(64, activation='relu')(input_sbj)
        hidden_sbj = layers.Dense(64, activation='relu')(hidden_sbj)
        output_sbj = layers.Dense(self.embeddings_dim, activation='relu')(hidden_sbj)        
        # Rete neurale per l'oggetto
        input_obj = keras.Input(shape=(self.input_dim,))
        hidden_obj = layers.Dense(64, activation='relu')(input_obj)
        hidden_obj = layers.Dense(64, activation='relu')(hidden_obj)
        output_obj = layers.Dense(self.embeddings_dim, activation='relu')(hidden_obj)
        # Classificatore
        concat = layers.concatenate([output_sbj, output_obj])
        hidden = layers.Dense(128, activation='relu')(concat)
        hidden = layers.Dense(128, activation='relu')(hidden)
        output = layers.Dense(self.output_dim, activation='softmax')(hidden)
        # Model
        self.model = keras.Model(inputs=[input_sbj, input_obj], outputs=[output])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
        self.model.summary()


    # Addestra il modello 
    def train_model(self, dataset_folder):
        self.load_dataset(dataset_folder)
        # Genera esempi negativi (not_related) per test e validation set
        self.gen_neg_samples(self.train_input1, self.train_output, self.train_input2)
        self.gen_neg_samples(self.valid_input1, self.valid_output, self.valid_input2)
        # Codifica input ed output in matrici utilizzabili da Keras
        self.data_encode('train')
        self.data_encode('valid')
        self.data_encode('test')
        # Crea il modello in keras
        self.build_model()
        # Addestramento
        self.model.fit(x=[self.train_input1, self.train_input2], 
                       y=self.train_output,
                       batch_size=64, 
                       validation_data=([self.valid_input1, self.valid_input2], self.valid_output), 
                       epochs=20, verbose=1)
        # Test prestazioni 
        loss, acc = self.model.evaluate(x=[self.test_input1, self.test_input2], y=self.test_output)
        print('Loss:', loss, 'Accuracy:', acc)
        # Eliminazione strutture dati superflue
        #TODO        

    
    # Carica il modello #TODO
    def load_model(self, model_path, entity_map, relation_map):
        self.entity_map = 'mappa entità caricata'
        self.relation_map = 'mappa relazioni caricata'
        self.model = 'modello caricato'


    # Predizioni
    def predict_links(self, entity1, entity2):
        return 'relation'

    




## RUNNING AREA ##
if __name__ == '__main__':

    lp_model = KerasLinkPredictor()
    lp_model.train_model('dataset')


    #for k, v in list(lp_model.relation_map.items())[:10]:
        #print(k, '-->', v)
    #print(lp_model.test_input1)
    #print(getattr(lp_model, 'test_input1'))
    #print()
    #print(getattr(lp_model, 'valid_input1'))
    

    
