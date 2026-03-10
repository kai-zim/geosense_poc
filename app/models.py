"""
Pydantic v2 Modelle für GeoSense Standort-Assistent.
Input-Validierung + Output-Schemas.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from enum import Enum
import uuid
from datetime import datetime


# ============================================================================
# ENUMS FÜR DROPDOWN-AUSWAHL
# ============================================================================

class Vertical(str, Enum):
    TANKSTELLE = "tankstelle"
    RETAIL = "retail"
    GASTRO = "gastro"
    FITNESS = "fitness"
    DROGERIE = "drogerie"
    BAECKEREI = "baeckerei"

class StoreFormat(str, Enum):
    KLEIN = "klein"
    STANDARD = "standard"
    FLAGSHIP = "flagship"

class Oeffnungszeiten(str, Enum):
    STANDARD = "standard"         # 8-20 Uhr
    ERWEITERT = "erweitert"       # 6-22 Uhr
    RUND_UM_DIE_UHR = "24_7"      # 24/7

class TageszeitFokus(str, Enum):
    MORGENS = "morgens"
    MITTAGS = "mittags"
    ABENDS = "abends"
    NACHT = "nacht"
    ALLE = "alle"

class WetterSensitivitaet(str, Enum):
    NIEDRIG = "niedrig"
    MITTEL = "mittel"
    HOCH = "hoch"

class Saisonprofil(str, Enum):
    SOMMER = "sommer"
    WINTER = "winter"
    EVENTS = "events"
    NEUTRAL = "neutral"

class Optimierungsziel(str, Enum):
    UMSATZ_MAX = "umsatz_max"
    ROI_MAX = "roi_max"
    RISIKO_MIN = "risiko_min"
    PAYBACK_MIN = "payback_min"

class Risikoappetit(str, Enum):
    KONSERVATIV = "konservativ"
    NEUTRAL = "neutral"
    AGGRESSIV = "aggressiv"

class Preset(str, Enum):
    BALANCED = "balanced"
    GROWTH = "growth"
    LOW_RISK = "low_risk"
    CUSTOM = "custom"


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class GrunddatenInput(BaseModel):
    """Step 1: Grunddaten & Suchgebiet"""
    vertical: Vertical = Field(default=Vertical.TANKSTELLE, description="Branche/Vertikal")
    stadt_plz: str = Field(default="Berlin", min_length=2, max_length=100, description="Stadt oder PLZ für Suche")
    radius_km: float = Field(default=10.0, ge=1.0, le=50.0, description="Suchradius in km")
    isochrone_minuten: int = Field(default=10, ge=5, le=30, description="Erreichbarkeit in Minuten")
    kandidaten_anzahl: int = Field(default=200, ge=10, le=1000, description="Anzahl zu generierender Kandidaten")
    top_n: int = Field(default=10, ge=3, le=50, description="Top-N Ergebnisse anzeigen")
    seed: Optional[int] = Field(default=None, description="Random Seed für Reproduzierbarkeit")


class GeschaeftsmodellInput(BaseModel):
    """Step 2 A: Geschäftsmodell-Parameter"""
    store_format: StoreFormat = Field(default=StoreFormat.STANDARD)
    oeffnungszeiten: Oeffnungszeiten = Field(default=Oeffnungszeiten.STANDARD)
    flaeche_qm: float = Field(default=150.0, ge=20.0, le=5000.0, description="Verkaufsfläche in m²")
    miete_pro_qm: float = Field(default=25.0, ge=5.0, le=200.0, description="Miete €/m²/Monat")
    capex_euro: float = Field(default=150000.0, ge=10000.0, le=5000000.0, description="Investitionskosten €")
    opex_monat: float = Field(default=15000.0, ge=1000.0, le=500000.0, description="Betriebskosten €/Monat")
    zielkunden: list[str] = Field(
        default=["pendler", "familien"],
        description="Zielkunden: pendler, familien, touristen, studierende, premium"
    )


class NachfrageMobilitaetInput(BaseModel):
    """Step 2 B: Nachfrage & Mobilität"""
    gewicht_fussgaenger: float = Field(default=0.3, ge=0.0, le=1.0)
    gewicht_pendler: float = Field(default=0.3, ge=0.0, le=1.0)
    gewicht_drive_by: float = Field(default=0.2, ge=0.0, le=1.0)
    tageszeit_fokus: TageszeitFokus = Field(default=TageszeitFokus.ALLE)


class KaufkraftDemografieInput(BaseModel):
    """Step 2 C: Kaufkraft & Demografie"""
    gewicht_kaufkraft: float = Field(default=0.4, ge=0.0, le=1.0)
    altersgruppe_18_25: float = Field(default=0.25, ge=0.0, le=1.0, description="Fit 18-25 Jahre")
    altersgruppe_26_40: float = Field(default=0.35, ge=0.0, le=1.0, description="Fit 26-40 Jahre")
    altersgruppe_41_65: float = Field(default=0.30, ge=0.0, le=1.0, description="Fit 41-65 Jahre")
    altersgruppe_65_plus: float = Field(default=0.10, ge=0.0, le=1.0, description="Fit 65+ Jahre")


class WettbewerbInput(BaseModel):
    """Step 2 D: Wettbewerb & Kannibalisierung"""
    konkurrenz_radius_km: float = Field(default=2.0, ge=0.5, le=10.0)
    wettbewerber_typen: list[str] = Field(
        default=["direkt"],
        description="direkt, indirekt, online"
    )
    mindestabstand_eigene_km: float = Field(default=1.0, ge=0.0, le=10.0)
    gewicht_kannibalisierung: float = Field(default=0.3, ge=0.0, le=1.0)


class StandortInfrastrukturInput(BaseModel):
    """Step 2 E: Standort & Infrastruktur"""
    parkplaetze_min: int = Field(default=5, ge=0, le=500)
    oepnv_naehe_max_min: int = Field(default=10, ge=1, le=30, description="Max Gehminuten zu ÖPNV")
    sichtbarkeit_ecke: bool = Field(default=False, description="Ecklage bevorzugt")
    sichtbarkeit_highstreet: bool = Field(default=True, description="Hauptstraße bevorzugt")
    e_ladepunkte_erforderlich: bool = Field(default=False)


class WetterSaisonInput(BaseModel):
    """Step 2 F: Wetter & Saisonalität"""
    wetter_sensitivitaet: WetterSensitivitaet = Field(default=WetterSensitivitaet.MITTEL)
    saisonprofil: Saisonprofil = Field(default=Saisonprofil.NEUTRAL)


class ZieleRisikoInput(BaseModel):
    """Step 2 G: Ziele & Risiko"""
    optimierungsziel: Optimierungsziel = Field(default=Optimierungsziel.ROI_MAX)
    mindest_roi_prozent: float = Field(default=15.0, ge=0.0, le=100.0)
    max_payback_monate: int = Field(default=36, ge=6, le=120)
    risikoappetit: Risikoappetit = Field(default=Risikoappetit.NEUTRAL)


class ModellOptionen(BaseModel):
    """Step 2 H: Modell-Presets & Optionen"""
    preset: Preset = Field(default=Preset.BALANCED)
    explainability: bool = Field(default=True, description="Warum empfohlen anzeigen")
    sensitivitaetsanalyse: bool = Field(default=False, description="Ranking-Stabilität anzeigen")
    zeige_baseline_vergleich: bool = Field(default=False, description="Vergleich zu Baselines")


class AnalyzeInput(BaseModel):
    """Gesamter Input für /analyze Endpoint"""
    grunddaten: GrunddatenInput = Field(default_factory=GrunddatenInput)
    geschaeftsmodell: GeschaeftsmodellInput = Field(default_factory=GeschaeftsmodellInput)
    nachfrage: NachfrageMobilitaetInput = Field(default_factory=NachfrageMobilitaetInput)
    kaufkraft: KaufkraftDemografieInput = Field(default_factory=KaufkraftDemografieInput)
    wettbewerb: WettbewerbInput = Field(default_factory=WettbewerbInput)
    infrastruktur: StandortInfrastrukturInput = Field(default_factory=StandortInfrastrukturInput)
    wetter: WetterSaisonInput = Field(default_factory=WetterSaisonInput)
    ziele: ZieleRisikoInput = Field(default_factory=ZieleRisikoInput)
    optionen: ModellOptionen = Field(default_factory=ModellOptionen)


# ============================================================================
# OUTPUT SCHEMAS
# ============================================================================

class ScoreBreakdown(BaseModel):
    """Aufschlüsselung des Scores nach Kategorien"""
    nachfrage_score: float
    kaufkraft_score: float
    wettbewerb_score: float
    infrastruktur_score: float
    demografie_fit: float
    saisonalitaet_score: float


class RisikoTreiber(BaseModel):
    """Risikofaktoren eines Standorts"""
    konkurrenz_risiko: float
    mietniveau_risiko: float
    datenabdeckung_risiko: float
    kannibalisierung_risiko: float
    gesamt_risiko: float


class StandortErgebnis(BaseModel):
    """Ein einzelner Standort im Ergebnis"""
    rang: int
    adresse: str
    latitude: float
    longitude: float
    erwarteter_umsatz: float
    risiko: float
    confidence: float
    roi_prozent: float
    payback_monate: float
    datenabdeckung_prozent: float
    top_3_gruende: list[str]
    score_breakdown: ScoreBreakdown
    risiko_treiber: RisikoTreiber
    gesamtscore: float
    

class BaselineVergleich(BaseModel):
    """Vergleich GeoSense vs Baselines"""
    methode: str
    avg_umsatz: float
    avg_roi: float
    avg_risiko: float


class SensitivitaetsErgebnis(BaseModel):
    """Ergebnis der Sensitivitätsanalyse"""
    ranking_stabilitaet: float  # 0-1, wie stabil ist Top-N bei ±10% Gewichtsvariation
    kritische_parameter: list[str]


class AnalyzeOutput(BaseModel):
    """Gesamter Output von /analyze"""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_zusammenfassung: dict
    standorte: list[StandortErgebnis]
    baseline_vergleich: Optional[list[BaselineVergleich]] = None
    sensitivitaet: Optional[SensitivitaetsErgebnis] = None
    total_kandidaten: int
    gefiltert_count: int


# ============================================================================
# PRESET KONFIGURATIONEN
# ============================================================================

PRESET_CONFIGS = {
    Preset.BALANCED: {
        "beschreibung": "Ausgewogenes Profil für mittelfristiges Wachstum",
        "ziele": {"optimierungsziel": Optimierungsziel.ROI_MAX, "risikoappetit": Risikoappetit.NEUTRAL},
        "nachfrage": {"gewicht_fussgaenger": 0.3, "gewicht_pendler": 0.3, "gewicht_drive_by": 0.2},
        "kaufkraft": {"gewicht_kaufkraft": 0.4},
        "wettbewerb": {"gewicht_kannibalisierung": 0.3},
    },
    Preset.GROWTH: {
        "beschreibung": "Aggressives Wachstum, höheres Risiko akzeptiert",
        "ziele": {"optimierungsziel": Optimierungsziel.UMSATZ_MAX, "risikoappetit": Risikoappetit.AGGRESSIV},
        "nachfrage": {"gewicht_fussgaenger": 0.4, "gewicht_pendler": 0.35, "gewicht_drive_by": 0.25},
        "kaufkraft": {"gewicht_kaufkraft": 0.5},
        "wettbewerb": {"gewicht_kannibalisierung": 0.15},
    },
    Preset.LOW_RISK: {
        "beschreibung": "Konservativ, Fokus auf sichere Standorte",
        "ziele": {"optimierungsziel": Optimierungsziel.RISIKO_MIN, "risikoappetit": Risikoappetit.KONSERVATIV},
        "nachfrage": {"gewicht_fussgaenger": 0.25, "gewicht_pendler": 0.25, "gewicht_drive_by": 0.15},
        "kaufkraft": {"gewicht_kaufkraft": 0.35},
        "wettbewerb": {"gewicht_kannibalisierung": 0.5},
    }
}


# ============================================================================
# VERTICAL KONFIGURATIONEN
# ============================================================================

VERTICAL_CONFIG = {
    Vertical.TANKSTELLE: {
        "name": "Tankstelle",
        "baseline_umsatz": 120000,  # €/Monat
        "marge": 0.08,
        "typische_flaeche": 200,
        "wichtig": ["drive_by", "parkplaetze", "oeffnungszeiten"]
    },
    Vertical.RETAIL: {
        "name": "Retail/Einzelhandel",
        "baseline_umsatz": 80000,
        "marge": 0.25,
        "typische_flaeche": 300,
        "wichtig": ["fussgaenger", "kaufkraft", "sichtbarkeit"]
    },
    Vertical.GASTRO: {
        "name": "Gastronomie",
        "baseline_umsatz": 50000,
        "marge": 0.15,
        "typische_flaeche": 120,
        "wichtig": ["fussgaenger", "touristen", "tageszeit"]
    },
    Vertical.FITNESS: {
        "name": "Fitnessstudio",
        "baseline_umsatz": 40000,
        "marge": 0.35,
        "typische_flaeche": 800,
        "wichtig": ["pendler", "demografie", "parkplaetze"]
    },
    Vertical.DROGERIE: {
        "name": "Drogerie",
        "baseline_umsatz": 60000,
        "marge": 0.22,
        "typische_flaeche": 400,
        "wichtig": ["fussgaenger", "familien", "oepnv"]
    },
    Vertical.BAECKEREI: {
        "name": "Bäckerei",
        "baseline_umsatz": 25000,
        "marge": 0.40,
        "typische_flaeche": 60,
        "wichtig": ["fussgaenger", "morgens", "sichtbarkeit"]
    }
}
