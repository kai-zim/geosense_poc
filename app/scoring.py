"""
Scoring-Engine für GeoSense Standort-Assistent.
Berechnet Umsatz, Risiko, ROI, Payback und Confidence für Standorte.
"""
import numpy as np
from typing import Optional
from scipy.stats import kendalltau

from app.models import (
    AnalyzeInput, StandortErgebnis, ScoreBreakdown, RisikoTreiber,
    Vertical, VERTICAL_CONFIG, Optimierungsziel, Risikoappetit,
    TageszeitFokus, WetterSensitivitaet, Saisonprofil
)
from app.mock_data import StandortKandidat
from app.utils import normalize_minmax, weighted_average, clip_score, generate_gruende


def _estimate_own_distance_km(kandidat: StandortKandidat) -> float:
    """Leitet eine monotone Distanznäherung zum nächsten eigenen Standort ab."""
    if kandidat.eigene_standorte_naehe <= 0:
        return float("inf")

    return max(0.15, 2.5 / (kandidat.eigene_standorte_naehe + 1))


# ============================================================================
# HAUPT-SCORING FUNKTION
# ============================================================================

def score_kandidaten(
    kandidaten: list[StandortKandidat],
    inputs: AnalyzeInput
) -> list[StandortErgebnis]:
    """
    Bewertet alle Kandidaten basierend auf den Eingabeparametern.
    
    Args:
        kandidaten: Liste von Standort-Kandidaten
        inputs: Alle Eingabeparameter des Nutzers
    
    Returns:
        Liste von sortierten StandortErgebnis-Objekten
    """
    if not kandidaten:
        return []
    
    # Feature-Arrays für Batch-Verarbeitung
    n = len(kandidaten)
    
    # 1) Score-Breakdown für jeden Kandidaten berechnen
    breakdowns = []
    risiko_treiber_list = []
    
    for kandidat in kandidaten:
        breakdown = compute_score_breakdown(kandidat, inputs)
        breakdowns.append(breakdown)
        
        risiko_treiber = compute_risiko_treiber(kandidat, inputs)
        risiko_treiber_list.append(risiko_treiber)
    
    # 2) Gesamtscore berechnen
    gesamtscores = []
    for breakdown in breakdowns:
        gesamt = compute_gesamtscore(breakdown, inputs)
        gesamtscores.append(gesamt)
    
    # 3) Umsatz, ROI, Payback berechnen
    business_metrics = []
    for i, kandidat in enumerate(kandidaten):
        metrics = compute_business_metrics(
            kandidat, breakdowns[i], inputs
        )
        business_metrics.append(metrics)
    
    # 4) Confidence berechnen
    confidences = [
        compute_confidence(kandidaten[i], breakdowns[i])
        for i in range(n)
    ]
    
    scored_items = list(zip(
        range(n), gesamtscores, business_metrics, 
        breakdowns, risiko_treiber_list, confidences
    ))

    # 5) Finanzielle Mindestziele als harte Nebenbedingungen anwenden
    eligible_items = []
    for item in scored_items:
        _idx, _score, metrics, _breakdown, _risiko_t, _conf = item
        if metrics["roi"] < inputs.ziele.mindest_roi_prozent:
            continue
        if metrics["payback"] > inputs.ziele.max_payback_monate:
            continue
        eligible_items.append(item)

    if not eligible_items:
        eligible_items = scored_items
    
    # 6) Nach Optimierungsziel sortieren
    sorted_items = sort_by_objective(eligible_items, inputs.ziele.optimierungsziel)
    
    # 7) Ergebnisse erstellen
    ergebnisse = []
    for rang, (idx, score, metrics, breakdown, risiko_t, conf) in enumerate(sorted_items, 1):
        kandidat = kandidaten[idx]
        
        # Gründe generieren
        breakdown_dict = {
            "nachfrage_score": breakdown.nachfrage_score,
            "kaufkraft_score": breakdown.kaufkraft_score,
            "wettbewerb_score": breakdown.wettbewerb_score,
            "infrastruktur_score": breakdown.infrastruktur_score,
            "demografie_fit": breakdown.demografie_fit,
        }
        gruende = generate_gruende(breakdown_dict, top_n=3)
        
        ergebnis = StandortErgebnis(
            rang=rang,
            adresse=kandidat.adresse,
            latitude=kandidat.latitude,
            longitude=kandidat.longitude,
            erwarteter_umsatz=metrics["umsatz"],
            risiko=risiko_t.gesamt_risiko,
            confidence=conf,
            roi_prozent=metrics["roi"],
            payback_monate=metrics["payback"],
            datenabdeckung_prozent=kandidat.datenabdeckung * 100,
            top_3_gruende=gruende,
            score_breakdown=breakdown,
            risiko_treiber=risiko_t,
            gesamtscore=score
        )
        ergebnisse.append(ergebnis)
    
    return ergebnisse


def compute_score_breakdown(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> ScoreBreakdown:
    """
    Berechnet Score-Breakdown für einen Kandidaten.
    
    Args:
        kandidat: Der Standort-Kandidat
        inputs: Eingabeparameter
    
    Returns:
        ScoreBreakdown mit Einzelscores
    """
    # A) Nachfrage-Score
    nachfrage = compute_nachfrage_score(kandidat, inputs)
    
    # B) Kaufkraft-Score
    kaufkraft = compute_kaufkraft_score(kandidat, inputs)
    
    # C) Wettbewerb-Score (invertiert: weniger Konkurrenz = besser)
    wettbewerb = compute_wettbewerb_score(kandidat, inputs)
    
    # D) Infrastruktur-Score
    infrastruktur = compute_infrastruktur_score(kandidat, inputs)
    
    # E) Demografie-Fit
    demografie = compute_demografie_fit(kandidat, inputs)
    
    # F) Saisonalität
    saisonalitaet = compute_saisonalitaet_score(kandidat, inputs)
    
    return ScoreBreakdown(
        nachfrage_score=nachfrage,
        kaufkraft_score=kaufkraft,
        wettbewerb_score=wettbewerb,
        infrastruktur_score=infrastruktur,
        demografie_fit=demografie,
        saisonalitaet_score=saisonalitaet
    )


def compute_nachfrage_score(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> float:
    """Berechnet Nachfrage-Score basierend auf Mobilität."""
    nachfrage_input = inputs.nachfrage
    
    # Normalisierte Werte (0-1)
    fuss_norm = min(1.0, kandidat.fussgaenger_pro_tag / 10000)
    pendler_norm = kandidat.pendler_anteil
    drive_by_norm = min(1.0, kandidat.drive_by_traffic / 5000)
    
    # Tageszeit-Bonus
    tageszeit_bonus = 1.0
    fokus = nachfrage_input.tageszeit_fokus
    if fokus != TageszeitFokus.ALLE:
        fokus_key = fokus.value
        if fokus_key in kandidat.tageszeit_verteilung:
            tageszeit_bonus = 0.8 + kandidat.tageszeit_verteilung[fokus_key] * 0.5
    
    # Gewichteter Score
    weighted_components = weighted_average(
        [fuss_norm, pendler_norm, drive_by_norm],
        [nachfrage_input.gewicht_fussgaenger,
         nachfrage_input.gewicht_pendler,
         nachfrage_input.gewicht_drive_by]
    )
    score = weighted_average([weighted_components, fuss_norm], [0.2, 0.8])
    
    return clip_score(score * tageszeit_bonus, 0, 1)


def compute_kaufkraft_score(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> float:
    """Berechnet Kaufkraft-Score."""
    # Kaufkraft-Index: 100 = Durchschnitt
    # Normalisieren auf 0-1 mit 60-140 als Bereich
    kk_norm = (kandidat.kaufkraft_index - 60) / 80
    kk_norm = max(0, min(1, kk_norm))
    
    # Gewichtung aus Input
    gewicht = inputs.kaufkraft.gewicht_kaufkraft
    
    return clip_score(kk_norm * gewicht + (1 - gewicht) * 0.5, 0, 1)


def compute_wettbewerb_score(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> float:
    """
    Berechnet Wettbewerb-Score.
    Höher = weniger Konkurrenz = besser.
    """
    wettbewerb_input = inputs.wettbewerb
    radius = wettbewerb_input.konkurrenz_radius_km
    
    # Konkurrenten im relevanten Radius
    if radius <= 0.5:
        konkurrenten = kandidat.konkurrenten_500m
    elif radius <= 1.0:
        konkurrenten = kandidat.konkurrenten_1km
    else:
        konkurrenten = kandidat.konkurrenten_2km
    
    # Konkurrenz-Penalty (nicht-linear)
    konkurrenz_penalty = 1.0 / (1 + 0.3 * konkurrenten)
    
    own_distance_km = _estimate_own_distance_km(kandidat)

    # Kannibalisierungs-Gewicht wirkt nun direkt auf eigene-Standorte-Penalty.
    kannib_gewicht = clip_score(wettbewerb_input.gewicht_kannibalisierung, 0.0, 1.0)
    own_near_penalty = 1.0 / (1.0 + 0.25 * kandidat.eigene_standorte_naehe)
    kannib_penalty = (1.0 - kannib_gewicht) + kannib_gewicht * own_near_penalty

    # Mindestabstand ausschließlich gegen eigene Standorte prüfen.
    if wettbewerb_input.mindestabstand_eigene_km <= 0:
        own_distance_factor = 1.0
    else:
        ratio = own_distance_km / wettbewerb_input.mindestabstand_eigene_km
        own_distance_factor = clip_score(ratio, 0.2, 1.0)

    score = konkurrenz_penalty * (0.97 + 0.03 * kannib_penalty) * (0.97 + 0.03 * own_distance_factor)
    
    return clip_score(score, 0, 1)


def compute_infrastruktur_score(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> float:
    """Berechnet Infrastruktur-Score."""
    infra = inputs.infrastruktur
    
    scores = []
    weights = []
    
    # Parkplätze
    park_score = min(1.0, kandidat.parkplaetze / max(1, infra.parkplaetze_min))
    scores.append(park_score)
    weights.append(0.25)
    
    # ÖPNV-Nähe
    oepnv_score = 1.0 if kandidat.oepnv_minuten <= infra.oepnv_naehe_max_min else 0.5
    scores.append(oepnv_score)
    weights.append(0.25)
    
    # Sichtbarkeit
    sicht_score = 0.5
    if infra.sichtbarkeit_ecke and kandidat.ist_ecklage:
        sicht_score += 0.25
    if infra.sichtbarkeit_highstreet and kandidat.ist_highstreet:
        sicht_score += 0.25
    scores.append(min(1.0, sicht_score))
    weights.append(0.3)
    
    # E-Ladepunkte
    if infra.e_ladepunkte_erforderlich:
        e_score = 1.0 if kandidat.hat_e_ladepunkte else 0.3
    else:
        e_score = 0.7 if kandidat.hat_e_ladepunkte else 0.5
    scores.append(e_score)
    weights.append(0.2)
    
    return weighted_average(scores, weights)


def compute_demografie_fit(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> float:
    """Berechnet Demografie-Fit."""
    demo_input = inputs.kaufkraft
    
    # Gewichteter Fit
    fit = (
        kandidat.demografie_18_25 * demo_input.altersgruppe_18_25 +
        kandidat.demografie_26_40 * demo_input.altersgruppe_26_40 +
        kandidat.demografie_41_65 * demo_input.altersgruppe_41_65 +
        kandidat.demografie_65_plus * demo_input.altersgruppe_65_plus
    )
    
    # Normalisieren (theoretisch 0-1, aber praktisch kleiner)
    return clip_score(fit * 2, 0, 1)


def compute_saisonalitaet_score(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> float:
    """Berechnet Saisonalitäts-Score."""
    wetter = inputs.wetter
    
    # Wetter-Sensitivität
    sens = wetter.wetter_sensitivitaet
    if sens == WetterSensitivitaet.NIEDRIG:
        wetter_faktor = 1.0
    elif sens == WetterSensitivitaet.MITTEL:
        wetter_faktor = 1.0 - kandidat.wetter_sensitivitaet * 0.15
    else:
        wetter_faktor = 1.0 - kandidat.wetter_sensitivitaet * 0.3
    
    # Saisonprofil
    saison = wetter.saisonprofil
    if saison == Saisonprofil.NEUTRAL:
        saison_faktor = 1.0
    else:
        saison_faktor = kandidat.saisonalitaet_faktor
    
    return clip_score(wetter_faktor * saison_faktor, 0, 1)


def compute_gesamtscore(
    breakdown: ScoreBreakdown,
    inputs: AnalyzeInput
) -> float:
    """
    Berechnet gewichteten Gesamtscore.
    
    Args:
        breakdown: Score-Breakdown
        inputs: Eingabeparameter
    
    Returns:
        Gesamtscore 0-100
    """
    # Standard-Gewichte
    weights = {
        "nachfrage": 0.25,
        "kaufkraft": 0.20,
        "wettbewerb": 0.20,
        "infrastruktur": 0.15,
        "demografie": 0.10,
        "saisonalitaet": 0.10
    }
    
    # Anpassung nach Risikoappetit
    appetit = inputs.ziele.risikoappetit
    if appetit == Risikoappetit.KONSERVATIV:
        weights["wettbewerb"] = 0.30
        weights["nachfrage"] = 0.20
    elif appetit == Risikoappetit.AGGRESSIV:
        weights["nachfrage"] = 0.35
        weights["wettbewerb"] = 0.10
    
    # Score berechnen
    score = (
        breakdown.nachfrage_score * weights["nachfrage"] +
        breakdown.kaufkraft_score * weights["kaufkraft"] +
        breakdown.wettbewerb_score * weights["wettbewerb"] +
        breakdown.infrastruktur_score * weights["infrastruktur"] +
        breakdown.demografie_fit * weights["demografie"] +
        breakdown.saisonalitaet_score * weights["saisonalitaet"]
    )
    
    return clip_score(score * 100, 0, 100)


def compute_risiko_treiber(
    kandidat: StandortKandidat,
    inputs: AnalyzeInput
) -> RisikoTreiber:
    """
    Berechnet Risiko-Treiber für einen Standort.
    
    Args:
        kandidat: Der Standort-Kandidat
        inputs: Eingabeparameter
    
    Returns:
        RisikoTreiber mit Einzelrisiken
    """
    # Konkurrenz-Risiko
    konkurrenz_risiko = min(100, kandidat.konkurrenten_1km * 15)
    
    # Mietniveau-Risiko (hohe Miete = höheres Risiko)
    durchschnitt_miete = 25  # Annahme
    miete_abweichung = (kandidat.miete_pro_qm - durchschnitt_miete) / durchschnitt_miete
    mietniveau_risiko = clip_score(50 + miete_abweichung * 50, 0, 100)
    
    # Datenabdeckung-Risiko (wenig Daten = höheres Risiko)
    datenabdeckung_risiko = (1 - kandidat.datenabdeckung) * 100
    
    # Kannibalisierungs-Risiko
    kannibalisierung_risiko = min(100, kandidat.eigene_standorte_naehe * 40)
    
    # Gesamt-Risiko (gewichtet)
    gesamt = weighted_average(
        [konkurrenz_risiko, mietniveau_risiko, datenabdeckung_risiko, kannibalisierung_risiko],
        [0.35, 0.25, 0.25, 0.15]
    )
    
    return RisikoTreiber(
        konkurrenz_risiko=konkurrenz_risiko,
        mietniveau_risiko=mietniveau_risiko,
        datenabdeckung_risiko=datenabdeckung_risiko,
        kannibalisierung_risiko=kannibalisierung_risiko,
        gesamt_risiko=gesamt
    )


def compute_business_metrics(
    kandidat: StandortKandidat,
    breakdown: ScoreBreakdown,
    inputs: AnalyzeInput
) -> dict:
    """
    Berechnet Business-Metriken: Umsatz, ROI, Payback.
    
    Args:
        kandidat: Der Standort-Kandidat
        breakdown: Score-Breakdown
        inputs: Eingabeparameter
    
    Returns:
        Dict mit umsatz, roi, payback
    """
    vertical = inputs.grunddaten.vertical
    vertical_cfg = VERTICAL_CONFIG[vertical]
    geschaeft = inputs.geschaeftsmodell
    
    # Baseline-Umsatz je nach Vertikal
    baseline = vertical_cfg["baseline_umsatz"]
    
    # Umsatz-Modell (kalibriert):
    # Statt eines harten Produkts vieler [0..1]-Faktoren verwenden wir einen
    # balancierten Business-Index mit realistischen Bandbreiten.
    demand_score = (breakdown.nachfrage_score + breakdown.demografie_fit) / 2
    purchasing_power = breakdown.kaufkraft_score
    competition_quality = breakdown.wettbewerb_score
    infrastruktur_quality = breakdown.infrastruktur_score
    seasonality_quality = breakdown.saisonalitaet_score

    business_index = weighted_average(
        [
            demand_score,
            purchasing_power,
            competition_quality,
            infrastruktur_quality,
            seasonality_quality,
        ],
        [0.30, 0.24, 0.20, 0.16, 0.10],
    )

    # Rechnet den Index in einen plausiblen Umsatz-Multiplikator um:
    # schwache Lagen bleiben unter Baseline, starke Lagen können darüber liegen.
    score_mult = 0.70 + business_index * 0.85  # ~0.70 .. 1.55

    # Mikro-Lagefaktor aus POI-Nähe und Sichtbarkeit
    micro_location_index = weighted_average(
        [
            kandidat.poi_naehe_score,
            1.0 if kandidat.ist_highstreet else 0.0,
            1.0 if kandidat.ist_ecklage else 0.0,
        ],
        [0.6, 0.25, 0.15],
    )
    location_mult = 0.90 + micro_location_index * 0.30  # ~0.90 .. 1.20

    # Saisonalität nicht nur als Penalty, sondern moderat symmetrisch
    seasonality_mult = 0.95 + seasonality_quality * 0.15  # ~0.95 .. 1.10
    
    # Format-Multiplikator
    format_mult = {
        "klein": 0.6,
        "standard": 1.0,
        "flagship": 1.5
    }.get(geschaeft.store_format.value, 1.0)
    
    # Öffnungszeiten-Multiplikator
    oeffnung_mult = {
        "standard": 1.0,
        "erweitert": 1.15,
        "24_7": 1.3
    }.get(geschaeft.oeffnungszeiten.value, 1.0)
    
    # Umsatz berechnen
    umsatz = (
        baseline
        * score_mult
        * location_mult
        * seasonality_mult
        * format_mult
        * oeffnung_mult
    )
    
    # Faktoren für Fläche
    typische_flaeche = vertical_cfg.get("typische_flaeche", 200)
    flaechen_ratio = geschaeft.flaeche_qm / typische_flaeche
    umsatz *= min(1.4, max(0.7, 0.7 + flaechen_ratio * 0.5))
    
    # Kosten
    marge = vertical_cfg["marge"]

    # Mietansatz: Eingabe ist Primärtreiber, lokale Miete kalibriert leicht mit
    effective_miete_qm = weighted_average(
        [geschaeft.miete_pro_qm, kandidat.miete_pro_qm],
        [0.75, 0.25]
    )
    miete_monat = effective_miete_qm * geschaeft.flaeche_qm

    # OPEX-Kalibrierung je Vertikal + Flächeneffekt
    vertical_opex_factor = {
        "tankstelle": 0.55,
        "retail": 0.95,
        "gastro": 0.80,
        "fitness": 0.70,
        "drogerie": 0.90,
        "baeckerei": 0.65,
    }.get(vertical.value, 1.0)
    size_cost_factor = min(1.4, max(0.6, 0.7 + flaechen_ratio * 0.5))

    opex = geschaeft.opex_monat * vertical_opex_factor * size_cost_factor + miete_monat
    capex = geschaeft.capex_euro
    
    # Gewinn
    profit_monat = umsatz * marge - opex
    profit_jahr = profit_monat * 12
    
    # ROI
    if capex > 0:
        roi = (profit_jahr / capex) * 100
    else:
        roi = 0
    
    # Payback
    if profit_monat > 0:
        payback = capex / profit_monat
    else:
        payback = 999  # Sehr lang
    
    return {
        "umsatz": max(0, umsatz),
        "roi": roi,
        "payback": min(payback, 120)  # Max 10 Jahre
    }


def compute_confidence(
    kandidat: StandortKandidat,
    breakdown: ScoreBreakdown
) -> float:
    """
    Berechnet Konfidenz-Score (0-100).
    
    Basiert auf:
    - Datenabdeckung
    - Score-Homogenität (keine extremen Ausreißer)
    """
    # Datenabdeckung (50% des Confidence)
    daten_conf = kandidat.datenabdeckung * 50
    
    # Score-Homogenität (50% des Confidence)
    scores = [
        breakdown.nachfrage_score,
        breakdown.kaufkraft_score,
        breakdown.wettbewerb_score,
        breakdown.infrastruktur_score
    ]
    std = np.std(scores)
    homogenitaet = 1.0 - min(1.0, std * 2)
    homo_conf = homogenitaet * 50
    
    return clip_score(daten_conf + homo_conf, 0, 100)


def sort_by_objective(
    items: list,
    objective: Optimierungsziel
) -> list:
    """
    Sortiert Items nach Optimierungsziel.
    
    Args:
        items: Liste von (idx, score, metrics, breakdown, risiko, conf)
        objective: Optimierungsziel
    
    Returns:
        Sortierte Liste
    """
    if objective == Optimierungsziel.UMSATZ_MAX:
        return sorted(items, key=lambda x: x[2]["umsatz"], reverse=True)
    elif objective == Optimierungsziel.ROI_MAX:
        return sorted(items, key=lambda x: x[2]["roi"], reverse=True)
    elif objective == Optimierungsziel.RISIKO_MIN:
        return sorted(items, key=lambda x: x[4].gesamt_risiko)
    elif objective == Optimierungsziel.PAYBACK_MIN:
        return sorted(items, key=lambda x: x[2]["payback"])
    else:
        # Default: Gesamtscore
        return sorted(items, key=lambda x: x[1], reverse=True)


# ============================================================================
# FILTERING
# ============================================================================

def filter_kandidaten(
    kandidaten: list[StandortKandidat],
    inputs: AnalyzeInput
) -> list[StandortKandidat]:
    """
    Filtert Kandidaten basierend auf Mindestanforderungen.
    
    Args:
        kandidaten: Alle Kandidaten
        inputs: Eingabeparameter
    
    Returns:
        Gefilterte Liste
    """
    filtered = []
    max_oepnv = min(
        inputs.infrastruktur.oepnv_naehe_max_min,
        inputs.grunddaten.isochrone_minuten,
    )
    
    for k in kandidaten:
        # Mindest-Parkplätze
        if k.parkplaetze < inputs.infrastruktur.parkplaetze_min:
            continue
        
        # Max ÖPNV-Entfernung
        if k.oepnv_minuten > max_oepnv:
            continue
        
        # E-Ladepunkte erforderlich
        if inputs.infrastruktur.e_ladepunkte_erforderlich and not k.hat_e_ladepunkte:
            continue
        
        # Mindestabstand zu eigenen Standorten
        own_distance_km = _estimate_own_distance_km(k)
        if own_distance_km < inputs.wettbewerb.mindestabstand_eigene_km:
            continue
        
        filtered.append(k)
    
    return filtered


# ============================================================================
# SENSITIVITÄTSANALYSE
# ============================================================================

def compute_sensitivity(
    kandidaten: list[StandortKandidat],
    inputs: AnalyzeInput,
    n_simulations: int = 20,
    variation: float = 0.1
) -> dict:
    """
    Führt Sensitivitätsanalyse durch.
    
    Variiert Gewichte um ±variation und misst Ranking-Stabilität.
    
    Args:
        kandidaten: Kandidaten-Liste
        inputs: Eingabeparameter
        n_simulations: Anzahl Simulationen
        variation: Prozentualer Variationsbereich
    
    Returns:
        Dict mit Stabilitäts-Metriken
    """
    if len(kandidaten) < 5:
        return {"ranking_stabilitaet": 1.0, "kritische_parameter": []}
    
    # Basis-Ranking über stabile Standort-Identität (Adresse) erstellen
    base_results = score_kandidaten(kandidaten, inputs)
    base_top_ids = [r.adresse for r in base_results[:10]]
    if not base_top_ids:
        return {"ranking_stabilitaet": 1.0, "kritische_parameter": []}
    
    rng = np.random.default_rng(42)
    kendall_taus = []
    
    for _ in range(n_simulations):
        # Gewichte variieren
        varied_inputs = inputs.model_copy(deep=True)
        
        # Variation der Nachfrage-Gewichte
        varied_inputs.nachfrage.gewicht_fussgaenger *= (1 + rng.uniform(-variation, variation))
        varied_inputs.nachfrage.gewicht_pendler *= (1 + rng.uniform(-variation, variation))
        varied_inputs.kaufkraft.gewicht_kaufkraft *= (1 + rng.uniform(-variation, variation))
        varied_inputs.wettbewerb.gewicht_kannibalisierung *= (1 + rng.uniform(-variation, variation))
        
        # Neues Ranking
        varied_results = score_kandidaten(kandidaten, varied_inputs)
        varied_top_ids = [r.adresse for r in varied_results[:10]]

        # Erst Identitäts-Overlap, dann Reihenfolge der gemeinsamen IDs.
        common_ids = [sid for sid in base_top_ids if sid in varied_top_ids]
        overlap = len(common_ids) / len(base_top_ids)

        if len(common_ids) >= 2:
            base_order = [base_top_ids.index(sid) for sid in common_ids]
            varied_order = [varied_top_ids.index(sid) for sid in common_ids]
            tau, _ = kendalltau(base_order, varied_order)
            tau_norm = 0.0 if np.isnan(tau) else (tau + 1.0) / 2.0
            kendall_taus.append(float(tau_norm * overlap))
        else:
            kendall_taus.append(0.0)
    
    if kendall_taus:
        stabilitaet = float(np.mean(kendall_taus))
    else:
        stabilitaet = 1.0
    
    # Kritische Parameter identifizieren (vereinfacht)
    kritische = []
    if stabilitaet < 0.8:
        kritische = ["Kaufkraft-Gewicht", "Wettbewerbs-Gewicht"]
    
    return {
        "ranking_stabilitaet": max(0, stabilitaet),
        "kritische_parameter": kritische
    }
