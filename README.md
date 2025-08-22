# AGAT: Automated Group Assembly Tool

Dieses Repository enthält den Quellcode und die Evaluationsdateien für die Bachelorarbeit "AGAT: Ein heuristischer Algorithmus zur präferenzbasierten Gruppenbildung mit fester Gruppengrösse" von Patrick Rüegsegger.

## Worum geht es?

Der hier präsentierte AGAT-Algorithmus ist ein heuristischer Lösungsansatz, der agglomeratives hierarchisches Clustering (AHC) für eine initiale Cluster-Bildung nutzt. Da AHC allein keine festen Gruppengrössen garantiert, wird der Algorithmus durch zwei neu entwickelte Reparaturstrategien ("merge" und "break") ergänzt, um diese strikte Anforderung zu erfüllen. Ein Präferenzmodell erlaubt dabei die Abgleichung sowohl auf Ähnlichkeit (homogene Kriterien) als auch auf Unterschiedlichkeit (heterogene Kriterien) der Gruppenmitglieder.


## Installation

1. Installieren Sie [Python 3.12.6](https://www.python.org/downloads/release/python-3126/)
 2. Klonen Sie das Git-Repository
	```
	git clone https://github.com/SEG-UNIBE/AGAT.git
	cd AGAT
	```
 3. Installieren Sie die Dependencies:
	```
	pip install -r requirements.txt
	```

## Anwendung

### Prototyp ausführen
Zur eigener Gruppenbildung können Sie den AGAT-Prototyp mit der grafischen Benutzeroberfläche starten, indem Sie die `main.py` ausführen:
```
python src/main.py
```
Wählen Sie anschliessend die `.csv` und `.json` Input-Dateien aus, die Sie für die Gruppenbildung benutzen möchten. Im Ordner `/examples` existieren bereits welche, die direkt genutzt oder angepasst werden können.

Nachdem Sie die Gruppenbildung ausgeführt haben, können Sie das Resultat als `.json`-Datei abspeichern.


### Evaluation reproduzieren

Das Skript `evaluation.py` führt die komplette Evaluation durch, wie sie in der Thesis beschrieben ist.

**Achtung:** Die vollständige Ausführung des Skripts dauert mehrere Stunden oder sogar wenigen Tagen (je nach Rechenleistung).
```
python evaluation/evaluations.py
```
Das Skript speichert die Ergebnisse standardmässig als `results.csv` im selben Verzeichnis.

## Zitation

Informationen zur Zitation dieser Arbeit finden Sie in der `CITATION.cff`-Datei.
