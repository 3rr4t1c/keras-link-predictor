# Benvenuto in KLIP (Keras LInk Predictor)
KLIP serve a predire i link mancanti in un grafo
***
Avrai bisogno di: Keras, Numpy, Tensorflow
***
#### Utilizzo:
- Si istanzia il modello con:
`klip = KerasLinkPredictor()`
- Come nella struttura di questo repository è richiesta una cartella con tre files.  
Per addestrare su un dataset:  
`klip.train(self, dataset_folder, input_dim=16, embeddings_dim=32, batch_size=64, epochs=20)`
- Per salvare il modello addestrato e le strutture necessarie:  
`klip.save('klip_save_test')`
- Per caricare il modello precedentemente salvato:  
`klip.load('klip_save_test.klip')`
- Per valutare l'accuracy del modello su un file:  
`klip.evaluate('dataset/test.tsv')`  
- Per effettuare predizioni:  
`p_map = klip.predict('dataset/test_instances.tsv', use_multiprocessing=False, workers=1)`
- Per effettuare predizioni singole:  
`klip.single_predict(self, entity1, entity2)`  
#### Note:
Le predizioni vengono restituite come `dict` di `dict` in cui una coppia di entità è mappata su un dizionario di relazioni mappate a loro volta sul relativo score. Nel caso servisse la matrice numpy originale si può usare direttamente il metodo `klip.model.predict` di Keras.  
Il risultato di una predizione ha la forma:  
`{('entity1', 'entity2'): {'birthPlace': 0.3, 'deathPlace': 0.45, 'spouse': 0.004}}`
