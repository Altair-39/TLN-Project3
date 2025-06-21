# Esercitazioni di Base

## Istruzioni per l'esecuzione dei programmi 

### Con pip

Per ogni esercitazione:

- Creare un ambiente virtuale e attivarlo

```bash

python -m venv venv

source venv/bin/activate

```

- Installare i requirements

```bash

pip install -r requirements.txt

```

### Con uv

Per ogni esercitazione:

```bash

uv run main.py

```

## Esercitazioni 

### Esercitazione 1 

L'obiettivo di questa esercitazione è quello di unire due risorse diverse. In questo caso la scelta è ricaduta su: 

- WordNet. 
- ConceptNet.

Nella nuova risorsa che si va a creare ogni parola presente in WordNet sarà associata alle corrispettive relazioni in ConceptNet.

### Esercitazione 2 

In questa esercitazione si ragiona sulla difficoltà nella creazione di buone definizioni. Si sono guardate due tipologie di similarità:

- Lessicale (SimLex): quanto le parole utilizzate nelle definizioni siano simili. 
- Semantica (SimSem): quanto le definizioni abbiano un senso comune.

Risultati ottenuti (medie):

- Pantalone:
    + SimLex: 0.1950
    + SimSem: 0.6093
- Microscopio:
    + SimLex: 0.1519
    + SimSem: 0.4734
- Pericolo:
    + SimLex: 0.1185
    + SimSem: 0.4783
- Euristica:
    + SimLex: 0.2276
    + SimSem: 0.0350

### Esercitazione 3

L'esercitazione 3 avrebbe dovuto prevedere un sistema per fare delle guess della parola partendo dalla sua definizione e usando il principio del genus differentia. Tuttavia è stato invece implementato un sistema per filtrare le definizioni migliori, ossia quelle che portano effettivemente al ritrovamento del termine in WordNet.

Risultati ottenuti (percentuali):

- Pantalone: 87.5%
- Microscopio: 56.4%
- Pericolo: 23.7%
- Euristica: 11.1%

### Esercitazione 4 

Nell'esercitazione 4 si è andati a esplorare il topic modelling con un dataset contenente 14.489 articoli medici e andando a produrre dei grafici per illustrare visivamente i topic ricavati. Si utilizzano cluster con minimo 10 elementi perché se se ne scelgono di più il sistema riconosce tutti i dati come outliers.

### Esercitazione 5

In quest'ultima esercitazione si è voluto sperimentare, con un modello di LLM, varie strategie di prompting. Gli obiettivi erano principalmente due: 

- Label the topic (dare un'etichetta ai topic dell'esercitazione 4): 
+ Zero-shot: invece che assegnare una label ai topic assume che tutti i concetti appartengano a un unico paziente. 
+ One-shot: con un esempio il task diventa più preciso riuscendo a riassumere le caratteristiche comuni in un’unica etichetta. 

- Guess from the Definitions (indovinare il termine a partire dalle definizioni delle esercitazioni 2 e 3):
+ Zero-shot: il modello fraintende il task assegnato e invece di restituire le parole restituisce una nuova definizione.
+ One-shot: aggiungendo un esempio sull’output atteso i risultati migliorano di molto. Il modello riesce a individuare correttamente le parole microscopio e pericolo, mentre è troppo generico su pantalone (tenta con indumento) ed euristica (linee guida).
+ One-shot with suggestions: aggiungendo al prompt precedente dei "suggerimenti", ossia le definizioni prese dal vocabolario della Treccani, si nota un miglioramento per la parola pantalone che adesso viene riconosciuta correttamente più spesso. Per quanto riguarda euristica viene restituito metodologia.
