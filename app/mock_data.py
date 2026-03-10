"""
Mock-Daten Generator für GeoSense.
Erzeugt synthetische Standort-Kandidaten mit realistischen Features.
Enthält eine versteckte "True Revenue" Funktion für Wirksamkeits-Tests.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from app.models import Vertical, VERTICAL_CONFIG
from app.utils import generate_address, calculate_distance_km


# ============================================================================
# STÄDTE-KOORDINATEN (Zentrum)
# ============================================================================

STADT_KOORDINATEN = {
    "berlin": (52.5200, 13.4050),
    "münchen": (48.1351, 11.5820),
    "hamburg": (53.5511, 9.9937),
    "frankfurt": (50.1109, 8.6821),
    "köln": (50.9375, 6.9603),
    "düsseldorf": (51.2277, 6.7735),
    "stuttgart": (48.7758, 9.1829),
    "leipzig": (51.3397, 12.3731),
    "dresden": (51.0504, 13.7373),
    "nürnberg": (49.4521, 11.0767),
}


@dataclass
class StandortKandidat:
    """Ein potentieller Standort mit allen Features."""
    id: int
    adresse: str
    latitude: float
    longitude: float
    entfernung_zentrum_km: float
    
    # Nachfrage & Mobilität
    fussgaenger_pro_tag: float
    pendler_anteil: float
    drive_by_traffic: float
    tageszeit_verteilung: dict  # morgens, mittags, abends, nacht
    
    # Kaufkraft & Demografie
    kaufkraft_index: float  # 100 = Durchschnitt
    demografie_18_25: float
    demografie_26_40: float
    demografie_41_65: float
    demografie_65_plus: float
    
    # Wettbewerb
    konkurrenten_500m: int
    konkurrenten_1km: int
    konkurrenten_2km: int
    naechster_konkurrent_m: float
    eigene_standorte_naehe: int
    
    # Infrastruktur
    parkplaetze: int
    oepnv_minuten: float
    ist_ecklage: bool
    ist_highstreet: bool
    hat_e_ladepunkte: bool
    
    # Miete & Kosten
    miete_pro_qm: float
    nebenkosten_faktor: float
    
    # Sonstige
    wetter_sensitivitaet: float  # 0-1, wie wetterabhängig
    saisonalitaet_faktor: float  # Multiplier je nach Saison
    poi_naehe_score: float  # Nähe zu interessanten Punkten
    datenabdeckung: float  # 0-1, wie vollständig die Daten sind
    
    # HIDDEN: True Revenue (für Experimente - nicht im normalen Scoring verwendet)
    _true_base_revenue: float = 0.0
    
    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary (ohne hidden fields)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def get_city_center(stadt_plz: str) -> tuple[float, float]:
    """
    Gibt Stadtzentrum-Koordinaten zurück.
    
    Args:
        stadt_plz: Stadtname oder PLZ
    
    Returns:
        (latitude, longitude)
    """
    stadt_lower = stadt_plz.lower().strip()
    
    # Direkte Suche
    if stadt_lower in STADT_KOORDINATEN:
        return STADT_KOORDINATEN[stadt_lower]
    
    # PLZ -> Default zu Berlin
    if stadt_lower.isdigit():
        # Einfache PLZ-Bereiche (vereinfacht)
        plz_int = int(stadt_lower[:2])
        if 10 <= plz_int <= 14:
            return STADT_KOORDINATEN["berlin"]
        elif 80 <= plz_int <= 85:
            return STADT_KOORDINATEN["münchen"]
        elif 20 <= plz_int <= 22:
            return STADT_KOORDINATEN["hamburg"]
        elif 60 <= plz_int <= 65:
            return STADT_KOORDINATEN["frankfurt"]
        elif 50 <= plz_int <= 51:
            return STADT_KOORDINATEN["köln"]
    
    # Default: Berlin
    return STADT_KOORDINATEN["berlin"]


def generate_candidates(
    stadt_plz: str,
    radius_km: float,
    anzahl: int,
    vertical: Vertical,
    seed: Optional[int] = None
) -> list[StandortKandidat]:
    """
    Generiert synthetische Standort-Kandidaten.
    
    Args:
        stadt_plz: Stadt oder PLZ als Zentrum
        radius_km: Suchradius in km
        anzahl: Anzahl zu generierender Kandidaten
        vertical: Branche für kontextspezifische Generierung
        seed: Random Seed für Reproduzierbarkeit
    
    Returns:
        Liste von StandortKandidat-Objekten
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    center_lat, center_lon = get_city_center(stadt_plz)
    vertical_cfg = VERTICAL_CONFIG[vertical]
    
    kandidaten = []
    
    for i in range(anzahl):
        # Zufällige Position im Radius (mit Bias zum Zentrum)
        # Verwende Rayleigh-Verteilung für realistischere Verteilung
        distance = rng.rayleigh(scale=radius_km / 2.5)
        distance = min(distance, radius_km)
        angle = rng.uniform(0, 2 * np.pi)
        
        # Konvertiere zu Koordinaten (vereinfacht)
        lat_offset = distance / 111.0 * np.cos(angle)
        lon_offset = distance / (111.0 * np.cos(np.radians(center_lat))) * np.sin(angle)
        
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        
        entfernung = calculate_distance_km(center_lat, center_lon, lat, lon)
        
        # Feature-Generierung mit räumlicher Korrelation
        # Näher am Zentrum = höhere Frequenz, aber auch höhere Miete
        zentrum_faktor = 1.0 - (entfernung / radius_km) * 0.7
        
        # Nachfrage korreliert mit Zentrumsnähe
        fussgaenger = int(rng.lognormal(mean=7 + zentrum_faktor, sigma=0.5))
        fussgaenger = min(max(fussgaenger, 100), 50000)
        
        pendler_anteil = rng.beta(2 + zentrum_faktor * 2, 5) 
        drive_by = rng.lognormal(mean=6 + (1 - zentrum_faktor), sigma=0.6)
        
        # Tageszeit-Verteilung
        tageszeit = {
            "morgens": rng.uniform(0.2, 0.4),
            "mittags": rng.uniform(0.2, 0.35),
            "abends": rng.uniform(0.2, 0.35),
            "nacht": rng.uniform(0.05, 0.15)
        }
        total = sum(tageszeit.values())
        tageszeit = {k: v/total for k, v in tageszeit.items()}
        
        # Kaufkraft (höher am Rand für Premium-Gebiete, aber auch in Zentrum)
        kaufkraft_base = 80 + zentrum_faktor * 30 + rng.normal(0, 15)
        kaufkraft_index = max(60, min(150, kaufkraft_base))
        
        # Demografie
        demografie_18_25 = rng.beta(2, 5)
        demografie_26_40 = rng.beta(3, 4)
        demografie_41_65 = rng.beta(3, 3)
        demografie_65_plus = 1.0 - demografie_18_25 - demografie_26_40 - demografie_41_65
        demografie_65_plus = max(0, demografie_65_plus)
        
        # Normalisieren
        total_demo = demografie_18_25 + demografie_26_40 + demografie_41_65 + demografie_65_plus
        if total_demo > 0:
            demografie_18_25 /= total_demo
            demografie_26_40 /= total_demo
            demografie_41_65 /= total_demo
            demografie_65_plus /= total_demo
        
        # Wettbewerb (mehr im Zentrum)
        konkurrenz_base = zentrum_faktor * 3
        konkurrenten_500m = int(rng.poisson(konkurrenz_base * 1.5))
        konkurrenten_1km = konkurrenten_500m + int(rng.poisson(konkurrenz_base * 2))
        konkurrenten_2km = konkurrenten_1km + int(rng.poisson(konkurrenz_base * 3))
        naechster_konkurrent = rng.exponential(500 / (1 + konkurrenten_500m))
        eigene_naehe = int(rng.poisson(0.5))
        
        # Infrastruktur
        parkplaetze = int(rng.poisson(15 * (1.5 - zentrum_faktor)))
        oepnv = rng.exponential(5) / zentrum_faktor if zentrum_faktor > 0.1 else 20
        oepnv = min(oepnv, 30)
        
        ist_ecklage = rng.random() < (0.15 + zentrum_faktor * 0.1)
        ist_highstreet = rng.random() < (0.1 + zentrum_faktor * 0.25)
        hat_e_lade = rng.random() < 0.15
        
        # Miete
        miete_base = 15 + zentrum_faktor * 35 + rng.normal(0, 5)
        miete = max(8, min(80, miete_base))
        nebenkosten = rng.uniform(1.1, 1.4)
        
        # Sonstige
        wetter_sens = rng.beta(2, 5)
        saison = rng.uniform(0.85, 1.15)
        poi_naehe = zentrum_faktor * 0.5 + rng.uniform(0, 0.5)
        datenabdeckung = rng.beta(8, 2)  # Meist hohe Abdeckung
        
        # ====================================================================
        # HIDDEN TRUE REVENUE FUNCTION
        # Dies ist die "wahre" Umsatz-Funktion, die für Baseline-Vergleiche
        # verwendet wird, aber nicht direkt im GeoSense-Scoring sichtbar ist.
        # ====================================================================
        true_revenue = compute_true_revenue(
            vertical=vertical,
            fussgaenger=fussgaenger,
            kaufkraft_index=kaufkraft_index,
            konkurrenten_1km=konkurrenten_1km,
            zentrum_faktor=zentrum_faktor,
            oepnv_minuten=oepnv,
            ist_highstreet=ist_highstreet,
            demografie_fit=demografie_26_40 + 0.5 * demografie_18_25,
            parkplaetze=parkplaetze,
            miete=miete,
            seed=seed + i if seed else i
        )
        
        kandidat = StandortKandidat(
            id=i,
            adresse=generate_address(seed + i if seed else i, stadt_plz),
            latitude=lat,
            longitude=lon,
            entfernung_zentrum_km=entfernung,
            fussgaenger_pro_tag=fussgaenger,
            pendler_anteil=pendler_anteil,
            drive_by_traffic=drive_by,
            tageszeit_verteilung=tageszeit,
            kaufkraft_index=kaufkraft_index,
            demografie_18_25=demografie_18_25,
            demografie_26_40=demografie_26_40,
            demografie_41_65=demografie_41_65,
            demografie_65_plus=demografie_65_plus,
            konkurrenten_500m=konkurrenten_500m,
            konkurrenten_1km=konkurrenten_1km,
            konkurrenten_2km=konkurrenten_2km,
            naechster_konkurrent_m=naechster_konkurrent,
            eigene_standorte_naehe=eigene_naehe,
            parkplaetze=parkplaetze,
            oepnv_minuten=oepnv,
            ist_ecklage=ist_ecklage,
            ist_highstreet=ist_highstreet,
            hat_e_ladepunkte=hat_e_lade,
            miete_pro_qm=miete,
            nebenkosten_faktor=nebenkosten,
            wetter_sensitivitaet=wetter_sens,
            saisonalitaet_faktor=saison,
            poi_naehe_score=poi_naehe,
            datenabdeckung=datenabdeckung,
            _true_base_revenue=true_revenue
        )
        
        kandidaten.append(kandidat)
    
    return kandidaten


def compute_true_revenue(
    vertical: Vertical,
    fussgaenger: float,
    kaufkraft_index: float,
    konkurrenten_1km: int,
    zentrum_faktor: float,
    oepnv_minuten: float,
    ist_highstreet: bool,
    demografie_fit: float,
    parkplaetze: int,
    miete: float,
    seed: int
) -> float:
    """
    DIE VERSTECKTE WAHRE UMSATZ-FUNKTION.
    
    Diese Funktion simuliert den "echten" Umsatz, den ein Standort
    generieren würde. Sie wird für Wirksamkeits-Experimente verwendet,
    um zu zeigen, dass GeoSense besser ist als Baselines.
    
    Die Funktion ist komplexer als einfache Heuristiken und enthält
    nicht-lineare Interaktionen, die naive Baselines nicht erfassen.
    """
    rng = np.random.default_rng(seed)
    
    vertical_cfg = VERTICAL_CONFIG[vertical]
    baseline = vertical_cfg["baseline_umsatz"]
    
    # === KOMPLEXE UMSATZ-FORMEL ===
    
    # 1) Nachfrage-Faktor (nicht-linear!)
    # Log-Transform, da Verdopplung der Frequenz nicht Verdopplung bringt
    freq_normalized = np.log1p(fussgaenger / 1000) / np.log1p(10)
    nachfrage_faktor = 0.5 + freq_normalized * 0.8
    
    # 2) Kaufkraft (mit Schwellenwert-Effekt)
    kaufkraft_faktor = 0.7 + 0.6 * (1 / (1 + np.exp(-(kaufkraft_index - 90) / 15)))
    
    # 3) Konkurrenz-Penalty (nicht-linear, starker Effekt bei vielen Konkurrenten)
    konkurrenz_penalty = 1.0 / (1 + 0.15 * konkurrenten_1km + 0.02 * konkurrenten_1km**2)
    
    # 4) Standort-Bonus
    standort_bonus = 1.0
    if ist_highstreet:
        standort_bonus += 0.15
    if zentrum_faktor > 0.7:
        standort_bonus += 0.1
    
    # 5) Erreichbarkeit
    oepnv_faktor = 1.0 - 0.02 * max(0, oepnv_minuten - 5)
    oepnv_faktor = max(0.7, oepnv_faktor)
    
    # 6) Demografie-Interaktion (je nach Vertical unterschiedlich wichtig)
    if vertical == Vertical.FITNESS:
        demo_faktor = 0.8 + demografie_fit * 0.5
    elif vertical == Vertical.DROGERIE:
        demo_faktor = 0.9 + demografie_fit * 0.2
    else:
        demo_faktor = 0.85 + demografie_fit * 0.3
    
    # 7) Vertikale-spezifische Faktoren
    if vertical == Vertical.TANKSTELLE:
        # Parkplätze und Drive-by wichtiger
        parking_bonus = min(1.3, 1.0 + parkplaetze * 0.01)
    elif vertical == Vertical.GASTRO:
        # Zentrum wichtiger
        parking_bonus = 1.0 + zentrum_faktor * 0.2
    else:
        parking_bonus = 1.0 + min(parkplaetze, 20) * 0.005
    
    # === KOMBINIEREN ===
    true_revenue = (
        baseline 
        * nachfrage_faktor 
        * kaufkraft_faktor 
        * konkurrenz_penalty 
        * standort_bonus 
        * oepnv_faktor 
        * demo_faktor 
        * parking_bonus
    )
    
    # === RAUSCHEN HINZUFÜGEN (realistische Varianz) ===
    noise_factor = rng.normal(1.0, 0.1)  # ±10% Rauschen
    true_revenue *= noise_factor
    
    return max(0, true_revenue)


def get_observed_revenue(true_revenue: float, seed: int) -> float:
    """
    Simuliert beobachteten Umsatz mit Mess-Rauschen.
    
    Args:
        true_revenue: Wahrer Umsatz
        seed: Random Seed
    
    Returns:
        Beobachteter Umsatz (mit Rauschen)
    """
    rng = np.random.default_rng(seed)
    
    # Heteroskedastisches Rauschen (mehr bei höheren Werten)
    noise_std = true_revenue * 0.15
    return max(0, true_revenue + rng.normal(0, noise_std))


def candidates_to_dataframe(kandidaten: list[StandortKandidat]) -> pd.DataFrame:
    """
    Konvertiert Kandidaten-Liste zu Pandas DataFrame.
    
    Args:
        kandidaten: Liste von StandortKandidat
    
    Returns:
        DataFrame mit allen Features
    """
    records = []
    for k in kandidaten:
        record = k.to_dict()
        # Tageszeit-Dict in Spalten aufteilen
        for key, val in k.tageszeit_verteilung.items():
            record[f"tageszeit_{key}"] = val
        del record["tageszeit_verteilung"]
        records.append(record)
    
    return pd.DataFrame(records)


# ============================================================================
# CACHE FÜR GENERIERTE DATEN
# ============================================================================

_candidates_cache: dict[str, list[StandortKandidat]] = {}


def get_or_generate_candidates(
    stadt_plz: str,
    radius_km: float,
    anzahl: int,
    vertical: Vertical,
    seed: Optional[int] = None
) -> list[StandortKandidat]:
    """
    Holt Kandidaten aus Cache oder generiert neue.
    
    Args:
        stadt_plz: Stadt oder PLZ
        radius_km: Suchradius
        anzahl: Anzahl Kandidaten
        vertical: Branche
        seed: Random Seed
    
    Returns:
        Liste von Kandidaten
    """
    cache_key = f"{stadt_plz}_{radius_km}_{anzahl}_{vertical.value}_{seed}"
    
    if cache_key not in _candidates_cache:
        _candidates_cache[cache_key] = generate_candidates(
            stadt_plz, radius_km, anzahl, vertical, seed
        )
    
    return _candidates_cache[cache_key]


def clear_candidates_cache():
    """Cache leeren."""
    global _candidates_cache
    _candidates_cache = {}
