# Keras Link Predictor (KLIP) by Enrico Verdolotti
# Link Prediction Home Made with Neural Networks :)
from os.path import join
import numpy as np
import pickle
import random
import keras
from keras import layers
import csv 

# Questo oggetto espone questi metodi per l'utilizzo:
# train_model: avvia un addestramento ed utilizza il modello appena addestrato
# load_model: carica un modello preesistente da disco e le strutture dati necessarie
# save_model: salva il modello corrente su disco e le strutture dati necessarie
# predict: effettua una predizione su una o più coppie di entità
class KerasLinkPredictor:
    
    # Inizializza il modello con dimensione input ed embeddings
    def __init__(self):        
        self.model_state = 'NOT_READY'


    # Carica il dataset nelle strutture necessarie
    def load_dataset(self, dataset_folder, input_dim):

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
            with open(join(dataset_folder, f), 'r', encoding='utf8') as tsv_file:
                reader = csv.reader(tsv_file, delimiter='\t')
                for line in reader:
                    h, r, t = line
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
                        self.entity_map[h] = np.random.rand(input_dim).tolist()
                    if t not in self.entity_map:
                        self.entity_map[t] = np.random.rand(input_dim).tolist()
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


    # Codifica i dati da liste di stringhe a matrici numpy
    def dataset_to_numpy(self, input1, input2, output):

        # Costruzione indice {(soggeto, oggetto): vettore multiclasse}
        h_index = dict()
        for i in range(len(input1)):
            sbj = input1[i]
            obj = input2[i]
            out = self.relation_map[output[i]]
            try:
                h_index[(sbj, obj)] += np.array(out)
            except:
                h_index[(sbj, obj)] = np.array(out)
        
        # Costruzione nuovo dataset a partire dall'indice
        in1_matrix, in2_matrix, out_matrix = [], [], []
        for k, multi_rel in h_index.items():
            sbj, obj = k        
            in1_matrix.append(np.array(self.entity_map[sbj]))
            in2_matrix.append(np.array(self.entity_map[obj]))
            out_matrix.append(multi_rel)

        # Conversione da liste di array a matrici numpy
        in1_matrix = np.vstack(in1_matrix)
        in2_matrix = np.vstack(in2_matrix)
        out_matrix = np.vstack(out_matrix)

        return in1_matrix, in2_matrix, out_matrix


    # Costruttore del modello    
    def build_model(self, input_dim, embeddings_dim):        
        # Rete neurale per il soggetto
        input_sbj = keras.Input(shape=(input_dim,))
        hidden_sbj = layers.Dense(64, activation='relu')(input_sbj)
        hidden_sbj = layers.Dense(64, activation='relu')(hidden_sbj)
        output_sbj = layers.Dense(embeddings_dim, activation='relu')(hidden_sbj)        
        # Rete neurale per l'oggetto
        input_obj = keras.Input(shape=(input_dim,))
        hidden_obj = layers.Dense(64, activation='relu')(input_obj)
        hidden_obj = layers.Dense(64, activation='relu')(hidden_obj)
        output_obj = layers.Dense(embeddings_dim, activation='relu')(hidden_obj)
        # Classificatore
        concat = layers.concatenate([output_sbj, output_obj])
        hidden = layers.Dense(128, activation='relu')(concat)
        hidden = layers.Dense(128, activation='relu')(hidden)
        output = layers.Dense(self.output_dim, activation='sigmoid')(hidden)
        
        # Model
        self.model = keras.Model(inputs=[input_sbj, input_obj], outputs=[output])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    # Addestra il modello e valuta le prestazioni 
    def train(self, dataset_folder, neg_samples_frac=0.8, input_dim=16, embeddings_dim=32, batch_size=64, epochs=20):

        # Caricamento dataset da disco # TODO: prevedere anche un passaggio per parametro?
        self.load_dataset(dataset_folder, input_dim)
        
        # Genera esempi negativi (not_related) per test e validation set
        self.gen_neg_samples(self.train_input1, self.train_output, self.train_input2, neg_samples_frac)
        self.gen_neg_samples(self.valid_input1, self.valid_output, self.valid_input2, neg_samples_frac)
        
        # Codifica input ed output in matrici utilizzabili da Keras        
        a, b, c = self.dataset_to_numpy(self.train_input1, self.train_input2, self.train_output)
        self.train_input1, self.train_input2, self.train_output = a, b, c        
        a, b, c = self.dataset_to_numpy(self.valid_input1, self.valid_input2, self.valid_output)
        self.valid_input1, self.valid_input2, self.valid_output = a, b, c        
        a, b, c = self.dataset_to_numpy(self.test_input1, self.test_input2, self.test_output)
        self.test_input1, self.test_input2, self.test_output = a, b, c        

        # Crea il modello in keras
        self.build_model(input_dim, embeddings_dim)
        
        # Addestramento
        self.model.fit(x=[self.train_input1, self.train_input2], 
                       y=self.train_output,
                       batch_size=batch_size, 
                       validation_data=([self.valid_input1, self.valid_input2], self.valid_output), 
                       epochs=epochs, 
                       verbose=1)      
        
        # Test finale prestazioni 
        loss, acc = self.model.evaluate(x=[self.test_input1, self.test_input2], y=self.test_output)
        print('Loss:', loss, 'Accuracy:', acc)        
        
        # Eliminazione strutture dati superflue
        del(self.train_input1, self.train_input2, self.train_output)
        del(self.valid_input1, self.valid_input2, self.valid_output)
        del(self.test_input1, self.test_input2, self.test_output)    

        # Modello pronto 
        self.model_state = 'READY'


    # Salva il modello su disco
    def save(self, file_name):

        assert self.model_state == 'READY', 'Nessun modello da salvare!'
        
        model_pack = {'model': self.model, 
                      'e_map': self.entity_map,
                      'r_map': self.relation_map}        

        with open(file_name + '.klip', 'wb') as model_file:
            pickle.dump(model_pack, model_file)
        

    # Carica il modello da disco
    def load(self, model_path):
        
        with open(model_path, 'rb') as model_file:
            model_pack = pickle.load(model_file)

        self.model = model_pack['model']
        self.entity_map = model_pack['e_map']
        self.relation_map = model_pack['r_map']


    ### Inizio metodi per utilizzare il modello ###

    # Valutazione modello, il file passato
    # deve essere un ipotetico sottografo del kg
    def evaluate(self, input_test):

        assert self.model_state == 'READY', 'Nessun modello da valutare!'

        # Se viene passato un path carica da file
        if type(input_test) is str:
            test = list()
            self.load_tsv(input_test, test)
        else:
            test = input_test

        in1, in2, out = [], [], []
        for h, r, t in test:
            in1.append(h)
            in2.append(t)
            out.append(r)

        in1, in2, out = self.dataset_to_numpy(in1, in2, out)
 
        loss, acc = self.model.evaluate(x=[in1, in2], y=out)
        print('Loss:', loss, 'Accuracy:', acc) 


    # Carica un file tsv da disco come lista di tuple (records)
    def load_tsv(self, file_path, target):

        with open(file_path, 'r', encoding='utf8') as tsv_file:
            rd = csv.reader(tsv_file, delimiter='\t')
            for line in rd:                
                target.append(tuple(line))


    # Costruisce una mappa che associa ad ogni istanza
    # gli score predetti per ogni relazione 
    # e.g. {('e1', 'e2'): {'birthPlace': 0.02, 'spouse': 0.7}}
    def build_output_map(self, instances, output):

        rel_names = list(self.relation_map.keys())
        output_map = dict()

        for i, pair in enumerate(instances):
            row = output[i, :]  # Riga dell'output, predizione per questa istanza
            rel_d = {rel_names[i]: row[i] for i in range(row.size)}
            output_map[pair] = rel_d

        return output_map 


    # Predice le relazioni per una lista di istanze
    # nel formato: [(entity1, entity2), (entity3, entity4), ...]
    # ciascuna entity deve essere nota al KG di addestramento.
    def predict(self, input_instances, use_multiprocessing=False, workers=1):
        
        # Se viene passato un path carica da file
        if type(input_instances) is str:
            instances = list()
            self.load_tsv(input_instances, instances)
        else:
            instances = input_instances
        
        # Costruzione numpy matrix per i due input
        input1, input2 = list(), list()
        for sbj, obj in instances:
            input1.append(self.entity_map[sbj])
            input2.append(self.entity_map[obj])
        input1 = np.array(input1)
        input2 = np.array(input2)

        # Effettua le predizoni, TODO: gestire multi-processing
        output = self.model.predict([input1, input2], 
                                    use_multiprocessing=use_multiprocessing, 
                                    workers=workers)

        # Converti in un formato più semplice da utilizzare
        return self.build_output_map(instances, output)


    # Effettua una singola predizione
    def single_predict(self, entity1, entity2):
        p = self.predict([(entity1, entity2)])
        return dict(p.values())
   


## TEST AREA ##
if __name__ == '__main__':

    klip = KerasLinkPredictor()    
    klip.train('datasets/LectorKG/500kSplit')
    #klip.save('klip_save_test')
    #klip.load('klip_save_test.klip')
    #klip.evaluate('datasets/test.tsv')

    #p_map = klip.predict('dataset/test_instances.tsv')    
    
    # for k, v in list(p_map.items())[:100]:
    #     max_rel = max(v, key=v.get)
    #     print(k, '->', max_rel, 'Score:', v[max_rel])

    
    

    
