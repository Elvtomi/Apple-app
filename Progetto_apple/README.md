
# 🍎 Apple Quality ML

Questo progetto permette di analizzare e predire la qualità delle mele utilizzando modelli di machine learning.  
L'applicazione è sviluppata con **Streamlit** e include:


- Caricamento di dataset CSV/XLSX
- Pulizia automatica dei dati
- Analisi esplorativa (EDA)
- Inferenza con modelli pre-addestrati (Random Forest e SVC)
- Visualizzazione della Confusion Matrix
- Download delle predizioni in CSV

Notebook di addestramento

Il notebook apple_notebook.ipynb mostra come sono stati addestrati i modelli:

- Pulizia e preprocessing dei dati
- Selezione delle feature principali: ['Size', 'Sweetness', 'Juiciness', 'Ripeness']
- Addestramento dei modelli
- Salvataggio dei modelli (.pkl) nella cartella models/

Nota: il notebook serve come riferimento o per riaddestrare i modelli, non è necessario per eseguire l’app.