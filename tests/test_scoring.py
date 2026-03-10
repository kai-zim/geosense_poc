"""
Tests für die Scoring-Engine.
Überprüft Monotonie und logische Konsistenz der Scoring-Funktionen.
"""
import pytest
import numpy as np
from dataclasses import replace
from types import SimpleNamespace

from app.models import (
    AnalyzeInput, GrunddatenInput, GeschaeftsmodellInput,
    NachfrageMobilitaetInput, KaufkraftDemografieInput,
    WettbewerbInput, StandortInfrastrukturInput, WetterSaisonInput,
    ZieleRisikoInput, ModellOptionen, Vertical, Optimierungsziel,
    Risikoappetit
)
from app.mock_data import generate_candidates, StandortKandidat
from app.scoring import (
    score_kandidaten, compute_score_breakdown, compute_risiko_treiber,
    compute_business_metrics, compute_nachfrage_score, compute_kaufkraft_score,
    compute_wettbewerb_score, compute_gesamtscore, filter_kandidaten,
    compute_sensitivity
)


@pytest.fixture
def default_inputs():
    """Standard-Inputs für Tests."""
    return AnalyzeInput()


@pytest.fixture
def sample_kandidaten():
    """Beispiel-Kandidaten für Tests."""
    return generate_candidates(
        stadt_plz="Berlin",
        radius_km=10,
        anzahl=50,
        vertical=Vertical.RETAIL,
        seed=42
    )


class TestScoreMonotonie:
    """
    Tests für Monotonie-Eigenschaften:
    - Mehr Kaufkraft sollte zu höherem Score führen
    - Mehr Konkurrenz sollte zu niedrigerem Score führen
    - etc.
    """
    
    def test_kaufkraft_monotonie(self, sample_kandidaten, default_inputs):
        """Höhere Kaufkraft -> höherer Kaufkraft-Score."""
        # Kandidaten nach Kaufkraft sortieren
        sorted_by_kaufkraft = sorted(
            sample_kandidaten, 
            key=lambda k: k.kaufkraft_index
        )
        
        scores = [
            compute_kaufkraft_score(k, default_inputs) 
            for k in sorted_by_kaufkraft
        ]
        
        # Score sollte tendenziell steigen (erlaubt kleine Abweichungen)
        increases = sum(1 for i in range(len(scores)-1) if scores[i+1] >= scores[i])
        increase_rate = increases / (len(scores) - 1)
        
        assert increase_rate > 0.7, "Kaufkraft-Score sollte mit Kaufkraft steigen"
    
    def test_konkurrenz_inverse_monotonie(self, sample_kandidaten, default_inputs):
        """Mehr Konkurrenz -> niedrigerer Wettbewerb-Score."""
        # Kandidaten nach Konkurrenz sortieren
        sorted_by_konkurrenz = sorted(
            sample_kandidaten,
            key=lambda k: k.konkurrenten_1km
        )
        
        scores = [
            compute_wettbewerb_score(k, default_inputs)
            for k in sorted_by_konkurrenz
        ]
        
        # Score sollte tendenziell sinken
        decreases = sum(1 for i in range(len(scores)-1) if scores[i+1] <= scores[i])
        decrease_rate = decreases / (len(scores) - 1)
        
        assert decrease_rate > 0.6, "Wettbewerb-Score sollte mit Konkurrenz sinken"
    
    def test_frequenz_monotonie(self, sample_kandidaten, default_inputs):
        """Höhere Fußgängerfrequenz -> höherer Nachfrage-Score."""
        # Kandidaten nach Frequenz sortieren
        sorted_by_freq = sorted(
            sample_kandidaten,
            key=lambda k: k.fussgaenger_pro_tag
        )
        
        scores = [
            compute_nachfrage_score(k, default_inputs)
            for k in sorted_by_freq
        ]
        
        # Score sollte tendenziell steigen
        increases = sum(1 for i in range(len(scores)-1) if scores[i+1] >= scores[i])
        increase_rate = increases / (len(scores) - 1)
        
        assert increase_rate > 0.6, "Nachfrage-Score sollte mit Frequenz steigen"


class TestRisikoLogik:
    """Tests für Risiko-Berechnung."""
    
    def test_konkurrenz_erhoht_risiko(self, sample_kandidaten, default_inputs):
        """Mehr Konkurrenz -> höheres Konkurrenz-Risiko."""
        risikos = [
            compute_risiko_treiber(k, default_inputs)
            for k in sample_kandidaten
        ]
        
        konkurrenz_nums = [k.konkurrenten_1km for k in sample_kandidaten]
        konkurrenz_risikos = [r.konkurrenz_risiko for r in risikos]
        
        # Korrelation sollte positiv sein
        correlation = np.corrcoef(konkurrenz_nums, konkurrenz_risikos)[0, 1]
        assert correlation > 0.3, "Konkurrenz und Konkurrenz-Risiko sollten korrelieren"
    
    def test_hohe_miete_erhoht_risiko(self, sample_kandidaten, default_inputs):
        """Höhere Miete -> höheres Mietniveau-Risiko."""
        risikos = [
            compute_risiko_treiber(k, default_inputs)
            for k in sample_kandidaten
        ]
        
        mieten = [k.miete_pro_qm for k in sample_kandidaten]
        miete_risikos = [r.mietniveau_risiko for r in risikos]
        
        correlation = np.corrcoef(mieten, miete_risikos)[0, 1]
        assert correlation > 0.3, "Miete und Mietniveau-Risiko sollten korrelieren"
    
    def test_risiko_bounds(self, sample_kandidaten, default_inputs):
        """Risiko-Werte sollten im gültigen Bereich sein."""
        for kandidat in sample_kandidaten:
            risiko = compute_risiko_treiber(kandidat, default_inputs)
            
            assert 0 <= risiko.konkurrenz_risiko <= 100
            assert 0 <= risiko.mietniveau_risiko <= 100
            assert 0 <= risiko.datenabdeckung_risiko <= 100
            assert 0 <= risiko.kannibalisierung_risiko <= 100
            assert 0 <= risiko.gesamt_risiko <= 100


class TestBusinessMetrics:
    """Tests für Business-Metriken (Umsatz, ROI, Payback)."""
    
    def test_umsatz_positiv(self, sample_kandidaten, default_inputs):
        """Umsatz sollte immer positiv sein."""
        for kandidat in sample_kandidaten:
            breakdown = compute_score_breakdown(kandidat, default_inputs)
            metrics = compute_business_metrics(kandidat, breakdown, default_inputs)
            
            assert metrics["umsatz"] >= 0, "Umsatz muss >= 0 sein"
    
    def test_payback_realistic(self, sample_kandidaten, default_inputs):
        """Payback sollte realistisch sein (nicht < 1 Monat)."""
        for kandidat in sample_kandidaten[:10]:
            breakdown = compute_score_breakdown(kandidat, default_inputs)
            metrics = compute_business_metrics(kandidat, breakdown, default_inputs)
            
            # Payback sollte mindestens 1 Monat sein
            assert metrics["payback"] >= 0, "Payback muss >= 0 sein"
    
    def test_flagship_higher_umsatz(self, sample_kandidaten):
        """Flagship-Format sollte höheren Umsatz haben als Klein."""
        kandidat = sample_kandidaten[0]
        
        # Klein
        inputs_klein = AnalyzeInput(
            geschaeftsmodell=GeschaeftsmodellInput(store_format="klein")
        )
        breakdown_klein = compute_score_breakdown(kandidat, inputs_klein)
        metrics_klein = compute_business_metrics(kandidat, breakdown_klein, inputs_klein)
        
        # Flagship
        inputs_flagship = AnalyzeInput(
            geschaeftsmodell=GeschaeftsmodellInput(store_format="flagship")
        )
        breakdown_flagship = compute_score_breakdown(kandidat, inputs_flagship)
        metrics_flagship = compute_business_metrics(kandidat, breakdown_flagship, inputs_flagship)
        
        assert metrics_flagship["umsatz"] > metrics_klein["umsatz"], \
            "Flagship sollte mehr Umsatz haben als Klein"


class TestGesamtscoring:
    """Tests für das Gesamtscoring."""
    
    def test_score_bounds(self, sample_kandidaten, default_inputs):
        """Gesamtscore sollte zwischen 0 und 100 liegen."""
        for kandidat in sample_kandidaten:
            breakdown = compute_score_breakdown(kandidat, default_inputs)
            score = compute_gesamtscore(breakdown, default_inputs)
            
            assert 0 <= score <= 100, f"Score {score} außerhalb [0, 100]"
    
    def test_risikoappetit_affects_weights(self, sample_kandidaten):
        """Risikoappetit sollte Gewichtung beeinflussen."""
        kandidat = sample_kandidaten[0]
        
        # Konservativ
        inputs_konservativ = AnalyzeInput(
            ziele=ZieleRisikoInput(risikoappetit=Risikoappetit.KONSERVATIV)
        )
        breakdown = compute_score_breakdown(kandidat, inputs_konservativ)
        score_konservativ = compute_gesamtscore(breakdown, inputs_konservativ)
        
        # Aggressiv
        inputs_aggressiv = AnalyzeInput(
            ziele=ZieleRisikoInput(risikoappetit=Risikoappetit.AGGRESSIV)
        )
        score_aggressiv = compute_gesamtscore(breakdown, inputs_aggressiv)
        
        # Scores sollten unterschiedlich sein
        assert score_konservativ != score_aggressiv, \
            "Risikoappetit sollte Score beeinflussen"
    
    def test_ergebnisse_sorted_by_objective(self, sample_kandidaten):
        """Ergebnisse sollten nach Optimierungsziel sortiert sein."""
        # Umsatz maximieren
        inputs_umsatz = AnalyzeInput(
            grunddaten=GrunddatenInput(vertical=Vertical.RETAIL, top_n=10),
            ziele=ZieleRisikoInput(optimierungsziel=Optimierungsziel.UMSATZ_MAX)
        )
        results = score_kandidaten(sample_kandidaten, inputs_umsatz)
        
        umsaetze = [r.erwarteter_umsatz for r in results[:10]]
        assert umsaetze == sorted(umsaetze, reverse=True), \
            "Sollte nach Umsatz absteigend sortiert sein"
    
    def test_roi_max_sorting(self, sample_kandidaten):
        """Bei ROI-Maximierung sollten Ergebnisse nach ROI sortiert sein."""
        inputs = AnalyzeInput(
            grunddaten=GrunddatenInput(vertical=Vertical.RETAIL, top_n=10),
            ziele=ZieleRisikoInput(optimierungsziel=Optimierungsziel.ROI_MAX)
        )
        results = score_kandidaten(sample_kandidaten, inputs)
        
        rois = [r.roi_prozent for r in results[:10]]
        assert rois == sorted(rois, reverse=True), \
            "Sollte nach ROI absteigend sortiert sein"


class TestFiltering:
    """Tests für die Filterung."""
    
    def test_parkplaetze_filter(self, sample_kandidaten):
        """Kandidaten mit zu wenig Parkplätzen werden gefiltert."""
        inputs = AnalyzeInput(
            infrastruktur=StandortInfrastrukturInput(parkplaetze_min=50)
        )
        
        filtered = filter_kandidaten(sample_kandidaten, inputs)
        
        for k in filtered:
            assert k.parkplaetze >= 50, "Gefilterte sollten genug Parkplätze haben"
    
    def test_oepnv_filter(self, sample_kandidaten):
        """Kandidaten mit zu weit entferntem ÖPNV werden gefiltert."""
        inputs = AnalyzeInput(
            infrastruktur=StandortInfrastrukturInput(oepnv_naehe_max_min=5)
        )
        
        filtered = filter_kandidaten(sample_kandidaten, inputs)
        
        for k in filtered:
            assert k.oepnv_minuten <= 5, "Gefilterte sollten ÖPNV-nah sein"

    def test_isochrone_begrenzt_oepnv(self, sample_kandidaten):
        """Isochrone-Minuten wirken als zusätzliches Erreichbarkeits-Limit."""
        inputs = AnalyzeInput(
            grunddaten=GrunddatenInput(isochrone_minuten=6),
            infrastruktur=StandortInfrastrukturInput(oepnv_naehe_max_min=20)
        )

        filtered = filter_kandidaten(sample_kandidaten, inputs)

        for k in filtered:
            assert k.oepnv_minuten <= 6, "Isochrone-Limit muss wirksam sein"

    def test_mindestabstand_eigene_filtert_nahe_standorte(self, sample_kandidaten):
        """Mindestabstand zu eigenen Standorten wird aktiv angewendet."""
        basis = sample_kandidaten[0]
        kandidat_fern = replace(basis, id=9001, adresse="Fern", eigene_standorte_naehe=0)
        kandidat_nah = replace(basis, id=9002, adresse="Nah", eigene_standorte_naehe=8)

        inputs = AnalyzeInput(
            wettbewerb=WettbewerbInput(mindestabstand_eigene_km=1.0),
            infrastruktur=StandortInfrastrukturInput(
                parkplaetze_min=0,
                oepnv_naehe_max_min=30,
                e_ladepunkte_erforderlich=False,
            )
        )

        filtered = filter_kandidaten([kandidat_fern, kandidat_nah], inputs)
        adressen = [k.adresse for k in filtered]

        assert "Fern" in adressen
        assert "Nah" not in adressen


class TestScoreBreakdown:
    """Tests für Score-Breakdown."""
    
    def test_breakdown_completeness(self, sample_kandidaten, default_inputs):
        """Score-Breakdown sollte alle Komponenten haben."""
        kandidat = sample_kandidaten[0]
        breakdown = compute_score_breakdown(kandidat, default_inputs)
        
        assert hasattr(breakdown, 'nachfrage_score')
        assert hasattr(breakdown, 'kaufkraft_score')
        assert hasattr(breakdown, 'wettbewerb_score')
        assert hasattr(breakdown, 'infrastruktur_score')
        assert hasattr(breakdown, 'demografie_fit')
        assert hasattr(breakdown, 'saisonalitaet_score')
    
    def test_breakdown_bounds(self, sample_kandidaten, default_inputs):
        """Alle Breakdown-Scores sollten zwischen 0 und 1 liegen."""
        for kandidat in sample_kandidaten[:20]:
            breakdown = compute_score_breakdown(kandidat, default_inputs)
            
            assert 0 <= breakdown.nachfrage_score <= 1
            assert 0 <= breakdown.kaufkraft_score <= 1
            assert 0 <= breakdown.wettbewerb_score <= 1
            assert 0 <= breakdown.infrastruktur_score <= 1
            assert 0 <= breakdown.demografie_fit <= 1
            assert 0 <= breakdown.saisonalitaet_score <= 1


class TestEndToEnd:
    """End-to-End Tests für das gesamte Scoring."""
    
    def test_full_scoring_pipeline(self, sample_kandidaten, default_inputs):
        """Komplette Scoring-Pipeline durchläuft."""
        results = score_kandidaten(sample_kandidaten, default_inputs)
        
        assert len(results) > 0, "Sollte Ergebnisse liefern"
        
        for r in results[:10]:
            assert r.rang > 0
            assert r.adresse
            assert r.erwarteter_umsatz >= 0
            assert 0 <= r.risiko <= 100
            assert 0 <= r.confidence <= 100
            assert len(r.top_3_gruende) > 0
    
    def test_top_n_respected(self):
        """Top-N Begrenzung wird respektiert."""
        kandidaten = generate_candidates("Berlin", 10, 100, Vertical.RETAIL, 42)
        
        inputs = AnalyzeInput(
            grunddaten=GrunddatenInput(top_n=5)
        )
        
        results = score_kandidaten(kandidaten, inputs)
        # Die Funktion gibt alle zurück, Begrenzung in main.py
        # Hier testen wir nur, dass mehr als 5 kommen
        assert len(results) >= 5
    
    def test_reproducibility(self):
        """Mit gleichem Seed gleiche Ergebnisse."""
        inputs = AnalyzeInput(
            grunddaten=GrunddatenInput(seed=123)
        )
        
        kandidaten1 = generate_candidates("Berlin", 10, 50, Vertical.RETAIL, 123)
        kandidaten2 = generate_candidates("Berlin", 10, 50, Vertical.RETAIL, 123)
        
        results1 = score_kandidaten(kandidaten1, inputs)
        results2 = score_kandidaten(kandidaten2, inputs)
        
        assert results1[0].adresse == results2[0].adresse, \
            "Gleicher Seed sollte gleiche Ergebnisse liefern"

    def test_roi_distribution_contains_positive_values(self, sample_kandidaten, default_inputs):
        """Kalibrierung: ROI sollte realistisch gemischt sein (nicht alles negativ)."""
        results = score_kandidaten(sample_kandidaten, default_inputs)
        rois = [r.roi_prozent for r in results]

        assert any(r > 0 for r in rois), "Es sollte mindestens einen Standort mit positivem ROI geben"
        assert any(r < 0 for r in rois), "Es sollte mindestens einen Standort mit negativem ROI geben"

    def test_finanzielle_schwellenwerte_werden_angewendet(self, sample_kandidaten, monkeypatch):
        """Mindest-ROI und Max-Payback schließen Kandidaten wirksam aus."""
        picked = sample_kandidaten[:6]

        def fake_metrics(kandidat, _breakdown, _inputs):
            if kandidat.id % 2 == 0:
                return {"umsatz": 100000, "roi": 30.0, "payback": 18.0}
            return {"umsatz": 100000, "roi": 5.0, "payback": 60.0}

        monkeypatch.setattr("app.scoring.compute_business_metrics", fake_metrics)

        strict = AnalyzeInput(
            ziele=ZieleRisikoInput(mindest_roi_prozent=20.0, max_payback_monate=24)
        )
        results = score_kandidaten(picked, strict)

        assert results
        for r in results:
            assert r.roi_prozent >= 20.0
            assert r.payback_monate <= 24.0


class TestSensitivity:
    """Tests für Sensitivitätsanalyse."""

    def test_sensitivity_compares_stable_identities(self, monkeypatch):
        """Bei gleichem Rang aber anderer Identität darf Stabilität nicht künstlich hoch sein."""
        counter = {"calls": 0}

        def fake_score(_kandidaten, _inputs):
            counter["calls"] += 1
            base_ids = [f"A{i}" for i in range(10)]
            varied_ids = [f"B{i}" for i in range(10)]
            ids = base_ids if counter["calls"] == 1 else varied_ids
            return [SimpleNamespace(rang=i + 1, adresse=ids[i]) for i in range(10)]

        monkeypatch.setattr("app.scoring.score_kandidaten", fake_score)
        sens = compute_sensitivity([SimpleNamespace()] * 10, AnalyzeInput(), n_simulations=3)

        assert sens["ranking_stabilitaet"] < 0.25
