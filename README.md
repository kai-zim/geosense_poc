# GeoSense Standort-Assistent

GeoSense ist ein Proof-of-Concept zur datenbasierten Bewertung potenzieller Filialstandorte. Die Anwendung kombiniert Standortfaktoren wie Nachfrage, Kaufkraft, Wettbewerb und Infrastruktur in einem nachvollziehbaren Scoring-Modell und stellt die Ergebnisse als Weboberfläche sowie als JSON-API bereit.

## Zielsetzung

Das Projekt unterstützt die vergleichende Bewertung mehrerer Standortkandidaten je Suchgebiet.

- Einheitliche Eingabe über ein mehrstufiges Formular oder JSON
- Ranking der Kandidaten nach Gesamt-Score
- Erklärbarkeit über Gründe pro Standort
- Optionaler Baseline-Vergleich und Sensitivitätsanalyse
- Export von Ergebnissen als CSV und JSON

## Funktionsumfang

- Unterstützte Vertikalen: Tankstelle, Retail, Gastro, Fitness, Drogerie, Bäckerei
- Presets: `balanced`, `growth`, `low_risk`
- Analysekanäle:
- HTML-Workflow über `POST /analyze`
- JSON-Workflow über `POST /analyze/json`
- Exporte:
- CSV über `GET /export/csv?run_id=...`
- JSON über `GET /export/json?run_id=...`

## Technologie-Stack

- Python 3.11+
- FastAPI
- Pydantic v2
- Jinja2 Templates
- NumPy, pandas, SciPy
- pytest

## Projektstruktur

```text
geosense_poc/
    app/
        main.py            FastAPI App und Endpunkte
        models.py          Datenmodelle und Validierung
        scoring.py         Scoring- und Ranking-Logik
        mock_data.py       Synthetische Standortdaten
        experiments.py     Baseline- und Wirksamkeitsexperimente
        utils.py           Hilfsfunktionen und Export
        templates/         HTML-Templates
        static/            CSS und JavaScript
    tests/
        test_main_api.py
        test_validation.py
        test_scoring.py
        test_experiments.py
    requirements.txt
    README.md
```

## Installation

```bash
cd geosense_poc
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

## Anwendung starten

```bash
uvicorn app.main:app --reload
```

Die Anwendung ist danach unter `http://127.0.0.1:8000` erreichbar.

## Tests

Alle Tests ausführen:

```bash
pytest -v
```

Einzelne Testmodule:

```bash
pytest tests/test_main_api.py -v
pytest tests/test_scoring.py -v
pytest tests/test_experiments.py -v
pytest tests/test_validation.py -v
```

## Wirksamkeitsexperimente

Die Datei `app/experiments.py` vergleicht GeoSense mit vier Baselines:

- Random
- Kaufkraft x Verkehr
- Innenstadt-Nähe
- Niedrige Miete

Zur schnellen Ausgabe eines Berichts:

```bash
python -c "from app.experiments import generate_wirksamkeits_report; print(generate_wirksamkeits_report(30))"
```

## API-Endpunkte

- `GET /` Hauptseite
- `POST /analyze` Standortanalyse über Formular
- `POST /analyze/json` Standortanalyse über JSON
- `GET /export/csv?run_id=...` CSV-Export eines Analyse-Laufs
- `GET /export/json?run_id=...` JSON-Export eines Analyse-Laufs
- `GET /health` Health-Check
- `GET /api/presets/{preset_name}` Preset-Konfiguration

## Hinweise

- Ergebnisse werden für Exporte in einem In-Memory-Cache gehalten.
- Das Projekt ist als akademischer Proof-of-Concept konzipiert.
