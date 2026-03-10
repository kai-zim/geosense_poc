"""
GeoSense Standort-Assistent - FastAPI Hauptanwendung.

Bietet eine Web-UI und API-Endpunkte für die Standortanalyse.
"""
import json
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models import (
    AnalyzeInput, AnalyzeOutput, GrunddatenInput, GeschaeftsmodellInput,
    NachfrageMobilitaetInput, KaufkraftDemografieInput, WettbewerbInput,
    StandortInfrastrukturInput, WetterSaisonInput, ZieleRisikoInput,
    ModellOptionen, Vertical, StoreFormat, Oeffnungszeiten, TageszeitFokus,
    WetterSensitivitaet, Saisonprofil, Optimierungsziel, Risikoappetit,
    Preset, PRESET_CONFIGS, VERTICAL_CONFIG, SensitivitaetsErgebnis,
    BaselineVergleich
)
from app.mock_data import get_or_generate_candidates, clear_candidates_cache
from app.scoring import score_kandidaten, filter_kandidaten, compute_sensitivity
from app.experiments import get_baseline_comparison_for_ui
from app.utils import results_to_csv, results_to_json_export


# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="GeoSense Standort-Assistent",
    description="KI-gestützte Standortanalyse für optimale Geschäftsentscheidungen",
    version="1.0.0"
)

# Pfade
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Static Files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-Memory Storage für Ergebnisse (für Export)
_results_cache: dict[str, dict] = {}


# ============================================================================
# HELPER FUNKTIONEN
# ============================================================================

def parse_form_to_input(form_data: dict) -> AnalyzeInput:
    """
    Konvertiert Form-Daten zu AnalyzeInput.
    
    Args:
        form_data: Dictionary mit Form-Feldern
    
    Returns:
        AnalyzeInput-Objekt
    """
    # Grunddaten
    grunddaten = GrunddatenInput(
        vertical=Vertical(form_data.get("vertical", "retail")),
        stadt_plz=form_data.get("stadt_plz", "Berlin"),
        radius_km=float(form_data.get("radius_km", 10)),
        isochrone_minuten=int(form_data.get("isochrone_minuten", 10)),
        kandidaten_anzahl=int(form_data.get("kandidaten_anzahl", 200)),
        top_n=int(form_data.get("top_n", 10)),
        seed=int(form_data.get("seed")) if form_data.get("seed") else None
    )
    
    # Zielkunden parsen
    zielkunden = form_data.getlist("zielkunden") if hasattr(form_data, 'getlist') else form_data.get("zielkunden", [])
    if isinstance(zielkunden, str):
        zielkunden = [zielkunden] if zielkunden else []
    
    # Geschäftsmodell
    geschaeftsmodell = GeschaeftsmodellInput(
        store_format=StoreFormat(form_data.get("store_format", "standard")),
        oeffnungszeiten=Oeffnungszeiten(form_data.get("oeffnungszeiten", "standard")),
        flaeche_qm=float(form_data.get("flaeche_qm", 150)),
        miete_pro_qm=float(form_data.get("miete_pro_qm", 25)),
        capex_euro=float(form_data.get("capex_euro", 150000)),
        opex_monat=float(form_data.get("opex_monat", 15000)),
        zielkunden=zielkunden if zielkunden else ["pendler", "familien"]
    )
    
    # Nachfrage
    nachfrage = NachfrageMobilitaetInput(
        gewicht_fussgaenger=float(form_data.get("gewicht_fussgaenger", 0.3)),
        gewicht_pendler=float(form_data.get("gewicht_pendler", 0.3)),
        gewicht_drive_by=float(form_data.get("gewicht_drive_by", 0.2)),
        tageszeit_fokus=TageszeitFokus(form_data.get("tageszeit_fokus", "alle"))
    )
    
    # Kaufkraft
    kaufkraft = KaufkraftDemografieInput(
        gewicht_kaufkraft=float(form_data.get("gewicht_kaufkraft", 0.4)),
        altersgruppe_18_25=float(form_data.get("altersgruppe_18_25", 0.25)),
        altersgruppe_26_40=float(form_data.get("altersgruppe_26_40", 0.35)),
        altersgruppe_41_65=float(form_data.get("altersgruppe_41_65", 0.30)),
        altersgruppe_65_plus=float(form_data.get("altersgruppe_65_plus", 0.10))
    )
    
    # Wettbewerber-Typen parsen
    wettbewerber = form_data.getlist("wettbewerber_typen") if hasattr(form_data, 'getlist') else form_data.get("wettbewerber_typen", [])
    if isinstance(wettbewerber, str):
        wettbewerber = [wettbewerber] if wettbewerber else []
    
    # Wettbewerb
    wettbewerb = WettbewerbInput(
        konkurrenz_radius_km=float(form_data.get("konkurrenz_radius_km", 2)),
        wettbewerber_typen=wettbewerber if wettbewerber else ["direkt"],
        mindestabstand_eigene_km=float(form_data.get("mindestabstand_eigene_km", 1)),
        gewicht_kannibalisierung=float(form_data.get("gewicht_kannibalisierung", 0.3))
    )
    
    # Infrastruktur
    infrastruktur = StandortInfrastrukturInput(
        parkplaetze_min=int(form_data.get("parkplaetze_min", 5)),
        oepnv_naehe_max_min=int(form_data.get("oepnv_naehe_max_min", 10)),
        sichtbarkeit_ecke=form_data.get("sichtbarkeit_ecke") == "on",
        sichtbarkeit_highstreet=form_data.get("sichtbarkeit_highstreet") == "on",
        e_ladepunkte_erforderlich=form_data.get("e_ladepunkte_erforderlich") == "on"
    )
    
    # Wetter
    wetter = WetterSaisonInput(
        wetter_sensitivitaet=WetterSensitivitaet(form_data.get("wetter_sensitivitaet", "mittel")),
        saisonprofil=Saisonprofil(form_data.get("saisonprofil", "neutral"))
    )
    
    # Ziele
    ziele = ZieleRisikoInput(
        optimierungsziel=Optimierungsziel(form_data.get("optimierungsziel", "roi_max")),
        mindest_roi_prozent=float(form_data.get("mindest_roi_prozent", 15)),
        max_payback_monate=int(form_data.get("max_payback_monate", 36)),
        risikoappetit=Risikoappetit(form_data.get("risikoappetit", "neutral"))
    )
    
    # Optionen
    optionen = ModellOptionen(
        preset=Preset(form_data.get("preset", "balanced")),
        explainability=form_data.get("explainability") == "on",
        sensitivitaetsanalyse=form_data.get("sensitivitaetsanalyse") == "on",
        zeige_baseline_vergleich=form_data.get("zeige_baseline_vergleich") == "on"
    )
    
    return AnalyzeInput(
        grunddaten=grunddaten,
        geschaeftsmodell=geschaeftsmodell,
        nachfrage=nachfrage,
        kaufkraft=kaufkraft,
        wettbewerb=wettbewerb,
        infrastruktur=infrastruktur,
        wetter=wetter,
        ziele=ziele,
        optionen=optionen
    )


def create_input_summary(inputs: AnalyzeInput) -> dict:
    """Erstellt eine Zusammenfassung der Eingaben."""
    return {
        "vertikal": VERTICAL_CONFIG[inputs.grunddaten.vertical]["name"],
        "suchgebiet": f"{inputs.grunddaten.stadt_plz}, {inputs.grunddaten.radius_km} km",
        "store_format": inputs.geschaeftsmodell.store_format.value,
        "optimierungsziel": inputs.ziele.optimierungsziel.value,
        "risikoappetit": inputs.ziele.risikoappetit.value,
        "preset": inputs.optionen.preset.value
    }


def build_optional_analysis_outputs(
    inputs: AnalyzeInput,
    gefiltert,
    top_ergebnisse,
) -> tuple[Optional[SensitivitaetsErgebnis], Optional[list[BaselineVergleich]]]:
    """Erstellt optionale Analyse-Erweiterungen konsistent für HTML und JSON."""
    sensitivitaet = None
    if inputs.optionen.sensitivitaetsanalyse:
        sens_result = compute_sensitivity(gefiltert, inputs)
        sensitivitaet = SensitivitaetsErgebnis(
            ranking_stabilitaet=sens_result["ranking_stabilitaet"],
            kritische_parameter=sens_result["kritische_parameter"],
        )

    baseline_vergleich = None
    if inputs.optionen.zeige_baseline_vergleich:
        id_zu_idx = {k.id: i for i, k in enumerate(gefiltert)}
        addr_zu_id = {k.adresse: k.id for k in gefiltert}
        geosense_indices = [
            id_zu_idx[addr_zu_id[e.adresse]]
            for e in top_ergebnisse
            if e.adresse in addr_zu_id and addr_zu_id[e.adresse] in id_zu_idx
        ]

        comparisons = get_baseline_comparison_for_ui(
            gefiltert, geosense_indices, inputs.grunddaten.top_n
        )
        baseline_vergleich = [
            BaselineVergleich(
                methode=c["methode"],
                avg_umsatz=c["avg_umsatz"],
                avg_roi=0,
                avg_risiko=0,
            )
            for c in comparisons
        ]

    return sensitivitaet, baseline_vergleich


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Hauptseite mit 3-Schritt-Formular."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "verticals": [(v.value, VERTICAL_CONFIG[v]["name"]) for v in Vertical],
            "store_formats": [(f.value, f.value.capitalize()) for f in StoreFormat],
            "oeffnungszeiten": [
                ("standard", "Standard (8-20 Uhr)"),
                ("erweitert", "Erweitert (6-22 Uhr)"),
                ("24_7", "24/7")
            ],
            "tageszeit_optionen": [(t.value, t.value.capitalize()) for t in TageszeitFokus],
            "wetter_optionen": [(w.value, w.value.capitalize()) for w in WetterSensitivitaet],
            "saison_optionen": [(s.value, s.value.capitalize()) for s in Saisonprofil],
            "ziel_optionen": [
                ("umsatz_max", "Umsatz maximieren"),
                ("roi_max", "ROI maximieren"),
                ("risiko_min", "Risiko minimieren"),
                ("payback_min", "Payback minimieren")
            ],
            "risiko_optionen": [(r.value, r.value.capitalize()) for r in Risikoappetit],
            "preset_optionen": [
                ("balanced", "Balanced - Ausgewogen"),
                ("growth", "Growth - Wachstum"),
                ("low_risk", "Low Risk - Sicherheit")
            ]
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request):
    """
    Führt die Standortanalyse durch.
    
    Nimmt Form-Daten, berechnet Scoring und gibt Ergebnisse zurück.
    """
    # Form-Daten parsen
    form = await request.form()
    form_dict = dict(form)
    
    # Mehrfachauswahl-Felder
    form_dict["zielkunden"] = form.getlist("zielkunden")
    form_dict["wettbewerber_typen"] = form.getlist("wettbewerber_typen")
    
    try:
        inputs = parse_form_to_input(form_dict)
    except Exception as e:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "error": f"Fehler bei der Eingabe-Validierung: {str(e)}"
            }
        )
    
    # Kandidaten generieren
    kandidaten = get_or_generate_candidates(
        stadt_plz=inputs.grunddaten.stadt_plz,
        radius_km=inputs.grunddaten.radius_km,
        anzahl=inputs.grunddaten.kandidaten_anzahl,
        vertical=inputs.grunddaten.vertical,
        seed=inputs.grunddaten.seed
    )
    
    # Filtern
    gefiltert = filter_kandidaten(kandidaten, inputs)
    
    # Scoring
    ergebnisse = score_kandidaten(gefiltert, inputs)
    
    # Top-N begrenzen
    top_ergebnisse = ergebnisse[:inputs.grunddaten.top_n]
    
    # Optionale Analyse-Erweiterungen
    sensitivitaet, baseline_vergleich = build_optional_analysis_outputs(
        inputs,
        gefiltert,
        top_ergebnisse,
    )
    
    # Output erstellen
    run_id = str(uuid.uuid4())[:8]
    output = AnalyzeOutput(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        input_zusammenfassung=create_input_summary(inputs),
        standorte=top_ergebnisse,
        baseline_vergleich=baseline_vergleich,
        sensitivitaet=sensitivitaet,
        total_kandidaten=len(kandidaten),
        gefiltert_count=len(gefiltert)
    )
    
    # In Cache speichern für Export
    _results_cache[run_id] = {
        "output": output.model_dump(),
        "inputs": inputs.model_dump()
    }
    
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "output": output,
            "show_explainability": inputs.optionen.explainability,
            "vertical_name": VERTICAL_CONFIG[inputs.grunddaten.vertical]["name"]
        }
    )


@app.post("/analyze/json")
async def analyze_json(inputs: AnalyzeInput):
    """
    API-Endpunkt für JSON-Eingabe.
    
    Args:
        inputs: AnalyzeInput-Objekt
    
    Returns:
        AnalyzeOutput als JSON
    """
    # Kandidaten generieren
    kandidaten = get_or_generate_candidates(
        stadt_plz=inputs.grunddaten.stadt_plz,
        radius_km=inputs.grunddaten.radius_km,
        anzahl=inputs.grunddaten.kandidaten_anzahl,
        vertical=inputs.grunddaten.vertical,
        seed=inputs.grunddaten.seed
    )
    
    # Filtern und Scoren
    gefiltert = filter_kandidaten(kandidaten, inputs)
    ergebnisse = score_kandidaten(gefiltert, inputs)
    top_ergebnisse = ergebnisse[:inputs.grunddaten.top_n]
    
    # Optionale Analyse-Erweiterungen
    sensitivitaet, baseline_vergleich = build_optional_analysis_outputs(
        inputs,
        gefiltert,
        top_ergebnisse,
    )

    # Output
    run_id = str(uuid.uuid4())[:8]
    output = AnalyzeOutput(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        input_zusammenfassung=create_input_summary(inputs),
        standorte=top_ergebnisse,
        baseline_vergleich=baseline_vergleich,
        sensitivitaet=sensitivitaet,
        total_kandidaten=len(kandidaten),
        gefiltert_count=len(gefiltert)
    )
    
    # Cache
    _results_cache[run_id] = {
        "output": output.model_dump(),
        "inputs": inputs.model_dump()
    }
    
    return output


@app.get("/export/csv")
async def export_csv(run_id: str = Query(..., description="Run ID der Analyse")):
    """
    Exportiert Ergebnisse als CSV.
    
    Args:
        run_id: ID des Analyse-Laufs
    
    Returns:
        CSV-Datei als Download
    """
    if run_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Run ID nicht gefunden")
    
    data = _results_cache[run_id]
    standorte = data["output"]["standorte"]
    
    csv_content = results_to_csv(standorte)
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=geosense_export_{run_id}.csv"
        }
    )


@app.get("/export/json")
async def export_json(run_id: str = Query(..., description="Run ID der Analyse")):
    """
    Exportiert Ergebnisse als JSON.
    
    Args:
        run_id: ID des Analyse-Laufs
    
    Returns:
        JSON-Datei als Download
    """
    if run_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Run ID nicht gefunden")
    
    data = _results_cache[run_id]
    export_data = results_to_json_export(data["output"])
    export_data["eingabeparameter"] = data["inputs"]
    
    json_content = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    return StreamingResponse(
        iter([json_content]),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=geosense_export_{run_id}.json"
        }
    )


@app.get("/health")
async def health():
    """Health Check Endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/presets/{preset_name}")
async def get_preset(preset_name: str):
    """
    Gibt Preset-Konfiguration zurück.
    
    Args:
        preset_name: Name des Presets
    
    Returns:
        Preset-Konfiguration als JSON
    """
    try:
        preset = Preset(preset_name)
        if preset in PRESET_CONFIGS:
            return PRESET_CONFIGS[preset]
        else:
            raise HTTPException(status_code=404, detail="Preset nicht gefunden")
    except ValueError:
        raise HTTPException(status_code=400, detail="Ungültiger Preset-Name")


# ============================================================================
# APP INIT
# ============================================================================

# Templates-Ordner erstellen falls nicht vorhanden
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
