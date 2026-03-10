"""
Hilfsfunktionen für GeoSense.
Normalisierung, Formatierung, Caching.
"""
import numpy as np
from typing import Callable, Any


def normalize_minmax(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Min-Max Normalisierung auf [0, 1].
    
    Args:
        values: Array von Werten
        invert: Wenn True, wird invertiert (höher = schlechter)
    
    Returns:
        Normalisiertes Array
    """
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_val - min_val < 1e-10:
        return np.ones_like(values) * 0.5
    
    normalized = (values - min_val) / (max_val - min_val)
    
    if invert:
        normalized = 1.0 - normalized
    
    return normalized


def normalize_zscore(values: np.ndarray) -> np.ndarray:
    """Z-Score Normalisierung."""
    mean = np.mean(values)
    std = np.std(values)
    
    if std < 1e-10:
        return np.zeros_like(values)
    
    return (values - mean) / std


def clip_score(score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Score auf Bereich begrenzen."""
    return max(min_val, min(max_val, score))


def format_currency(value: float, decimals: int = 0) -> str:
    """Formatiert Wert als Euro-Betrag."""
    if decimals == 0:
        return f"{int(value):,}".replace(",", ".") + " €"
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"


def format_percent(value: float, decimals: int = 1) -> str:
    """Formatiert Wert als Prozent."""
    return f"{value:.{decimals}f}%"


def format_months(value: float) -> str:
    """Formatiert Monate."""
    if value < 12:
        return f"{int(value)} Mon."
    years = value / 12
    return f"{years:.1f} Jahre"


def weighted_average(values: list[float], weights: list[float]) -> float:
    """
    Gewichteter Durchschnitt.
    
    Args:
        values: Liste von Werten
        weights: Liste von Gewichten (werden normalisiert)
    
    Returns:
        Gewichteter Durchschnitt
    """
    if not values or not weights:
        return 0.0
    
    total_weight = sum(weights)
    if total_weight < 1e-10:
        return sum(values) / len(values)
    
    normalized_weights = [w / total_weight for w in weights]
    return sum(v * w for v, w in zip(values, normalized_weights))


def generate_address(seed: int, city: str = "Berlin") -> str:
    """
    Generiert eine plausible Adresse für Mock-Daten.
    
    Args:
        seed: Seed für Reproduzierbarkeit
        city: Stadtname
    
    Returns:
        Adressstring
    """
    rng = np.random.default_rng(seed)
    
    strassen = [
        "Hauptstraße", "Bahnhofstraße", "Marktplatz", "Friedrichstraße",
        "Berliner Straße", "Goethestraße", "Schillerstraße", "Lindenallee",
        "Rosenweg", "Am Stadtpark", "Industriestraße", "Handelsweg",
        "Einkaufszentrum", "Neuer Markt", "Alte Landstraße", "Gewerbepark",
        "Stadtmitte", "Ringstraße", "Am Hafen", "Universitätsstraße"
    ]
    
    stadtteile = [
        "Zentrum", "Nord", "Süd", "Ost", "West",
        "Altstadt", "Neustadt", "Industriegebiet", "Gewerbegebiet",
        "Wohngebiet", "Einkaufsviertel", "Bahnhofsviertel"
    ]
    
    strasse = rng.choice(strassen)
    nummer = rng.integers(1, 200)
    stadtteil = rng.choice(stadtteile)
    
    return f"{strasse} {nummer}, {city}-{stadtteil}"


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Berechnet Entfernung in km (Haversine-Formel).
    
    Args:
        lat1, lon1: Koordinaten Punkt 1
        lat2, lon2: Koordinaten Punkt 2
    
    Returns:
        Entfernung in km
    """
    R = 6371  # Erdradius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def sigmoid(x: float, steepness: float = 1.0, midpoint: float = 0.0) -> float:
    """Sigmoid-Funktion für weiche Übergänge."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def decay_function(distance: float, decay_rate: float = 0.5) -> float:
    """
    Exponentieller Abfall mit Distanz.
    
    Args:
        distance: Entfernung
        decay_rate: Abfallrate (höher = schnellerer Abfall)
    
    Returns:
        Wert zwischen 0 und 1
    """
    return np.exp(-decay_rate * distance)


# ============================================================================
# CACHING
# ============================================================================

_cache: dict[str, Any] = {}


def cache_key(*args) -> str:
    """Generiert Cache-Key aus Argumenten."""
    return str(hash(str(args)))


def cached(func: Callable) -> Callable:
    """Einfacher Decorator für Caching."""
    def wrapper(*args, **kwargs):
        key = cache_key(func.__name__, args, tuple(sorted(kwargs.items())))
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]
    return wrapper


def clear_cache():
    """Cache leeren."""
    global _cache
    _cache = {}


def get_cache_size() -> int:
    """Cache-Größe zurückgeben."""
    return len(_cache)


# ============================================================================
# GRUND-GEBER
# ============================================================================

GRUENDE_TEMPLATES = {
    "hohe_frequenz": "Sehr hohe Passantenfrequenz ({value:.0f}/Tag)",
    "starke_kaufkraft": "Überdurchschnittliche Kaufkraft (Index: {value:.0f})",
    "geringe_konkurrenz": "Wenig direkte Konkurrenz im Umfeld ({value:.0f} Wettbewerber)",
    "gute_sichtbarkeit": "Hervorragende Sichtbarkeit (Ecklage/Hauptstraße)",
    "oepnv_anbindung": "Gute ÖPNV-Anbindung ({value:.0f} Min. zur Haltestelle)",
    "parkplaetze": "Ausreichend Parkplätze vorhanden ({value:.0f})",
    "demografie_fit": "Zielgruppen-Match von {value:.0f}%",
    "niedrige_miete": "Attraktives Mietniveau ({value:.0f} €/m²)",
    "wachstumsgebiet": "Standort in Wachstumsgebiet",
    "pendlerstrom": "Hoher Pendleranteil ({value:.0f}% der Frequenz)",
    "tourist_hotspot": "Touristisch attraktive Lage",
    "studentenviertel": "Nähe zu Universität/Hochschule",
}


def generate_gruende(score_breakdown: dict, top_n: int = 3) -> list[str]:
    """
    Generiert Top-N Gründe basierend auf Score-Breakdown.
    
    Args:
        score_breakdown: Dict mit Scores pro Kategorie
        top_n: Anzahl der Top-Gründe
    
    Returns:
        Liste von Grund-Strings
    """
    # Mapping von Score-Keys zu Grund-Templates
    score_to_grund = {
        "nachfrage_score": ("hohe_frequenz", 5000),
        "kaufkraft_score": ("starke_kaufkraft", 110),
        "wettbewerb_score": ("geringe_konkurrenz", 2),
        "infrastruktur_score": ("gute_sichtbarkeit", None),
        "demografie_fit": ("demografie_fit", 75),
    }
    
    # Scores sortieren
    sorted_scores = sorted(
        [(k, v) for k, v in score_breakdown.items() if k in score_to_grund],
        key=lambda x: x[1],
        reverse=True
    )
    
    gruende = []
    for key, score in sorted_scores[:top_n]:
        template_key, default_value = score_to_grund[key]
        template = GRUENDE_TEMPLATES.get(template_key, "")
        
        if "{value" in template and default_value is not None:
            # Skaliere Score zu plausiblem Wert
            scaled_value = default_value * (0.8 + score * 0.4)
            gruende.append(template.format(value=scaled_value))
        else:
            gruende.append(template.format(value=score * 100))
    
    return gruende


# ============================================================================
# EXPORT HELPERS
# ============================================================================

def results_to_csv(standorte: list[dict]) -> str:
    """
    Konvertiert Standort-Ergebnisse zu CSV-String.
    
    Args:
        standorte: Liste von Standort-Dicts
    
    Returns:
        CSV-String
    """
    if not standorte:
        return ""
    
    headers = [
        "Rang", "Adresse", "Erw. Umsatz (€)", "Risiko (%)", 
        "Confidence (%)", "ROI (%)", "Payback (Mon.)", 
        "Datenabdeckung (%)", "Gründe"
    ]
    
    lines = [";".join(headers)]
    
    for s in standorte:
        gruende = " | ".join(s.get("top_3_gruende", []))
        line = [
            str(s.get("rang", "")),
            s.get("adresse", ""),
            f"{s.get('erwarteter_umsatz', 0):.0f}",
            f"{s.get('risiko', 0):.1f}",
            f"{s.get('confidence', 0):.1f}",
            f"{s.get('roi_prozent', 0):.1f}",
            f"{s.get('payback_monate', 0):.1f}",
            f"{s.get('datenabdeckung_prozent', 0):.1f}",
            gruende
        ]
        lines.append(";".join(line))
    
    return "\n".join(lines)


def results_to_json_export(output: dict) -> dict:
    """
    Bereitet Output für JSON-Export auf.
    
    Args:
        output: AnalyzeOutput als Dict
    
    Returns:
        Bereingtes Dict für Export
    """
    return {
        "run_id": output.get("run_id"),
        "timestamp": output.get("timestamp"),
        "input": output.get("input_zusammenfassung"),
        "ergebnisse": [
            {
                "rang": s.get("rang"),
                "adresse": s.get("adresse"),
                "koordinaten": {"lat": s.get("latitude"), "lon": s.get("longitude")},
                "umsatz": s.get("erwarteter_umsatz"),
                "risiko": s.get("risiko"),
                "confidence": s.get("confidence"),
                "roi": s.get("roi_prozent"),
                "payback_monate": s.get("payback_monate"),
                "gruende": s.get("top_3_gruende")
            }
            for s in output.get("standorte", [])
        ],
        "meta": {
            "total_kandidaten": output.get("total_kandidaten"),
            "gefiltert": output.get("gefiltert_count")
        }
    }
