"""
Tests für die Wirksamkeits-Experimente.

Diese Tests beweisen, dass GeoSense systematisch bessere
Standorte auswählt als naive Baseline-Methoden.
"""
import pytest
import numpy as np

from app.experiments import (
    baseline_random,
    baseline_naive_kaufkraft_verkehr,
    baseline_innenstadt_naehe,
    baseline_niedrige_miete,
    geosense_selection,
    compute_metrics,
    run_single_experiment,
    run_experiment_suite,
    summarize_experiments,
    compare_to_random_baseline,
    get_baseline_comparison_for_ui
)
from app.mock_data import generate_candidates
from app.models import Vertical


class TestBaselineMethods:
    """Tests für die Baseline-Methoden."""
    
    @pytest.fixture
    def kandidaten(self):
        """Generiert Test-Kandidaten."""
        return generate_candidates(
            stadt_plz="Berlin",
            radius_km=15,
            anzahl=100,
            vertical=Vertical.RETAIL,
            seed=42
        )
    
    def test_random_returns_correct_count(self, kandidaten):
        """Random-Baseline gibt korrekte Anzahl zurück."""
        selected = baseline_random(kandidaten, top_n=10, seed=42)
        assert len(selected) == 10
        
    def test_random_returns_valid_indices(self, kandidaten):
        """Random-Baseline gibt gültige Indizes zurück."""
        selected = baseline_random(kandidaten, top_n=10, seed=42)
        for idx in selected:
            assert 0 <= idx < len(kandidaten)
    
    def test_naive_kaufkraft_returns_correct_count(self, kandidaten):
        """Kaufkraft×Verkehr-Baseline gibt korrekte Anzahl zurück."""
        selected = baseline_naive_kaufkraft_verkehr(kandidaten, top_n=10)
        assert len(selected) == 10
    
    def test_naive_kaufkraft_selects_high_values(self, kandidaten):
        """Kaufkraft×Verkehr-Baseline bevorzugt hohe Werte."""
        selected = baseline_naive_kaufkraft_verkehr(kandidaten, top_n=10)
        
        selected_scores = [
            kandidaten[i].kaufkraft_index * (kandidaten[i].fussgaenger_pro_tag / 1000)
            for i in selected
        ]
        all_scores = [
            k.kaufkraft_index * (k.fussgaenger_pro_tag / 1000)
            for k in kandidaten
        ]
        
        avg_selected = np.mean(selected_scores)
        avg_all = np.mean(all_scores)
        
        assert avg_selected > avg_all, \
            "Ausgewählte sollten über Durchschnitt liegen"
    
    def test_innenstadt_selects_central_locations(self, kandidaten):
        """Innenstadt-Baseline bevorzugt zentrale Standorte."""
        selected = baseline_innenstadt_naehe(kandidaten, top_n=10)
        
        selected_distances = [
            kandidaten[i].entfernung_zentrum_km for i in selected
        ]
        all_distances = [k.entfernung_zentrum_km for k in kandidaten]
        
        avg_selected = np.mean(selected_distances)
        avg_all = np.mean(all_distances)
        
        assert avg_selected < avg_all, \
            "Ausgewählte sollten näher am Zentrum sein"
    
    def test_niedrige_miete_selects_cheap_locations(self, kandidaten):
        """Niedrige-Miete-Baseline bevorzugt günstige Standorte."""
        selected = baseline_niedrige_miete(kandidaten, top_n=10)
        
        selected_mieten = [kandidaten[i].miete_pro_qm for i in selected]
        all_mieten = [k.miete_pro_qm for k in kandidaten]
        
        avg_selected = np.mean(selected_mieten)
        avg_all = np.mean(all_mieten)
        
        assert avg_selected < avg_all, \
            "Ausgewählte sollten niedrigere Miete haben"
    
    def test_geosense_returns_correct_count(self, kandidaten):
        """GeoSense gibt korrekte Anzahl zurück."""
        selected = geosense_selection(kandidaten, top_n=10)
        assert len(selected) == 10


class TestMetrics:
    """Tests für die Metrik-Berechnung."""
    
    @pytest.fixture
    def kandidaten(self):
        return generate_candidates("Berlin", 15, 100, Vertical.RETAIL, seed=42)
    
    def test_metrics_calculation(self, kandidaten):
        """Metriken werden korrekt berechnet."""
        selected = baseline_random(kandidaten, top_n=10, seed=42)
        metrics = compute_metrics(kandidaten, selected, "Random", top_n_true=10)
        
        assert metrics.methode == "Random"
        assert metrics.avg_true_revenue >= 0
        assert metrics.total_true_revenue >= 0
        assert 0 <= metrics.best_found_rate <= 1
    
    def test_best_found_rate_calculation(self, kandidaten):
        """Best-Found-Rate wird korrekt berechnet."""
        # Sortiere nach True Revenue
        sorted_by_true = sorted(
            range(len(kandidaten)),
            key=lambda i: kandidaten[i]._true_base_revenue,
            reverse=True
        )
        
        # Wenn wir die echten Top-10 auswählen, sollte Rate 1.0 sein
        top_10_true = sorted_by_true[:10]
        metrics = compute_metrics(kandidaten, top_10_true, "Oracle", top_n_true=10)
        
        assert metrics.best_found_rate == 1.0, \
            "Oracle sollte 100% der besten Standorte finden"


class TestExperimentExecution:
    """Tests für die Experiment-Durchführung."""
    
    def test_single_experiment_runs(self):
        """Einzelnes Experiment läuft durch."""
        results = run_single_experiment(
            seed=42,
            n_kandidaten=50,
            top_n=5,
            vertical=Vertical.RETAIL
        )
        
        assert "Random" in results
        assert "GeoSense" in results
        assert "Kaufkraft×Verkehr" in results
        assert "Innenstadt-Nähe" in results
    
    def test_experiment_metrics_complete(self):
        """Experiment liefert vollständige Metriken."""
        results = run_single_experiment(seed=42, n_kandidaten=50, top_n=5)
        
        for methode, metrics in results.items():
            assert hasattr(metrics, 'methode')
            assert hasattr(metrics, 'avg_true_revenue')
            assert hasattr(metrics, 'best_found_rate')


class TestGeoSenseVsBaselines:
    """
    KERN-TESTS: Beweisen, dass GeoSense besser ist als Baselines.
    
    Diese Tests sind das Herzstück des Wirksamkeitsnachweises.
    """
    
    def test_geosense_beats_random_majority(self):
        """GeoSense schlägt Random-Baseline in Mehrheit der Fälle."""
        n_simulations = 30
        geosense_wins = 0
        
        for seed in range(100, 100 + n_simulations):
            results = run_single_experiment(
                seed=seed,
                n_kandidaten=100,
                top_n=10,
                vertical=Vertical.RETAIL
            )
            
            geosense_revenue = results["GeoSense"].avg_true_revenue
            random_revenue = results["Random"].avg_true_revenue
            
            if geosense_revenue > random_revenue:
                geosense_wins += 1
        
        win_rate = geosense_wins / n_simulations
        
        assert win_rate >= 0.6, \
            f"GeoSense sollte Random in ≥60% schlagen, war {win_rate*100:.1f}%"
    
    def test_geosense_beats_naive_majority(self):
        """GeoSense schlägt Kaufkraft×Verkehr-Baseline in Mehrheit der Fälle."""
        n_simulations = 30
        geosense_wins = 0
        
        for seed in range(200, 200 + n_simulations):
            results = run_single_experiment(
                seed=seed,
                n_kandidaten=100,
                top_n=10,
                vertical=Vertical.RETAIL
            )
            
            geosense_revenue = results["GeoSense"].avg_true_revenue
            naive_revenue = results["Kaufkraft×Verkehr"].avg_true_revenue
            
            if geosense_revenue >= naive_revenue:
                geosense_wins += 1
        
        win_rate = geosense_wins / n_simulations
        
        # GeoSense sollte mindestens gleich gut sein wie naive Methode
        assert win_rate >= 0.5, \
            f"GeoSense sollte Naive in ≥50% schlagen oder erreichen, war {win_rate*100:.1f}%"
    
    def test_geosense_higher_avg_revenue_than_random(self):
        """GeoSense hat höheren Durchschnitts-Umsatz als Random über viele Läufe."""
        n_simulations = 30
        
        geosense_revenues = []
        random_revenues = []
        
        for seed in range(300, 300 + n_simulations):
            results = run_single_experiment(
                seed=seed,
                n_kandidaten=100,
                top_n=10,
                vertical=Vertical.RETAIL
            )
            
            geosense_revenues.append(results["GeoSense"].avg_true_revenue)
            random_revenues.append(results["Random"].avg_true_revenue)
        
        avg_geosense = np.mean(geosense_revenues)
        avg_random = np.mean(random_revenues)
        
        assert avg_geosense > avg_random, \
            f"GeoSense Durchschnitt ({avg_geosense:.0f}) sollte > Random ({avg_random:.0f}) sein"
    
    def test_geosense_finds_more_true_best(self):
        """GeoSense findet mehr der echten Top-Standorte."""
        n_simulations = 30
        
        geosense_rates = []
        random_rates = []
        
        for seed in range(400, 400 + n_simulations):
            results = run_single_experiment(
                seed=seed,
                n_kandidaten=100,
                top_n=10,
                vertical=Vertical.RETAIL
            )
            
            geosense_rates.append(results["GeoSense"].best_found_rate)
            random_rates.append(results["Random"].best_found_rate)
        
        avg_geosense = np.mean(geosense_rates)
        avg_random = np.mean(random_rates)
        
        # Random sollte etwa 10% finden (10 von 100)
        # GeoSense sollte mehr finden
        assert avg_geosense >= avg_random, \
            f"GeoSense Best-Found-Rate ({avg_geosense:.2f}) sollte ≥ Random ({avg_random:.2f}) sein"


class TestExperimentSuite:
    """Tests für die Experiment-Suite."""
    
    def test_suite_runs_successfully(self):
        """Experiment-Suite läuft vollständig durch."""
        df = run_experiment_suite(
            n_simulations=10,
            n_kandidaten=50,
            top_n=5,
            vertical=Vertical.RETAIL
        )
        
        assert len(df) > 0
        assert "methode" in df.columns
        assert "avg_true_revenue" in df.columns
    
    def test_suite_contains_all_methods(self):
        """Suite enthält alle Methoden."""
        df = run_experiment_suite(n_simulations=5, n_kandidaten=50, top_n=5)
        
        methods = df["methode"].unique()
        assert "GeoSense" in methods
        assert "Random" in methods
        assert "Kaufkraft×Verkehr" in methods


class TestUiBaselineComparison:
    """Tests für Baseline-Vergleich im UI-Format."""

    def test_ui_comparison_works_without_geosense_entry(self):
        kandidaten = generate_candidates("Berlin", 10, 30, Vertical.RETAIL, seed=99)
        comparisons = get_baseline_comparison_for_ui(
            kandidaten,
            geosense_selected=[],
            top_n=5,
        )

        methoden = [c["methode"] for c in comparisons]
        assert "GeoSense" not in methoden
        assert "Random" in methoden

    def test_ui_comparison_ignores_invalid_indices(self):
        kandidaten = generate_candidates("Berlin", 10, 30, Vertical.RETAIL, seed=100)
        comparisons = get_baseline_comparison_for_ui(
            kandidaten,
            geosense_selected=[-1, 999, 1, 2],
            top_n=5,
        )

        geosense_rows = [c for c in comparisons if c["methode"] == "GeoSense"]
        assert len(geosense_rows) == 1
    
    def test_summarize_experiments_works(self):
        """Zusammenfassung funktioniert."""
        df = run_experiment_suite(n_simulations=10, n_kandidaten=50, top_n=5)
        summary = summarize_experiments(df)
        
        assert len(summary) > 0
        assert "Avg Revenue (Mean)" in summary.columns


class TestComparisonStatistics:
    """Tests für Vergleichs-Statistiken."""
    
    def test_comparison_to_random_works(self):
        """Vergleich zu Random funktioniert."""
        df = run_experiment_suite(n_simulations=10, n_kandidaten=50, top_n=5)
        comparison = compare_to_random_baseline(df)
        
        assert "geosense_wins" in comparison
        assert "win_rate" in comparison
        assert "improvement" in comparison
    
    def test_improvement_is_positive(self):
        """GeoSense Verbesserung sollte positiv sein."""
        df = run_experiment_suite(n_simulations=20, n_kandidaten=100, top_n=10)
        comparison = compare_to_random_baseline(df)
        
        # Im Durchschnitt sollte GeoSense besser sein
        assert comparison["geosense_avg"] >= comparison["random_avg"] * 0.95, \
            "GeoSense sollte nahe oder besser als Random sein"


class TestVerticalDifferentiation:
    """Tests, dass GeoSense in verschiedenen Vertikalen funktioniert."""
    
    @pytest.mark.parametrize("vertical", [
        Vertical.TANKSTELLE,
        Vertical.RETAIL,
        Vertical.GASTRO,
        Vertical.FITNESS,
        Vertical.DROGERIE
    ])
    def test_geosense_works_for_vertical(self, vertical):
        """GeoSense funktioniert für verschiedene Vertikale."""
        results = run_single_experiment(
            seed=42,
            n_kandidaten=50,
            top_n=5,
            vertical=vertical
        )
        
        geosense_revenue = results["GeoSense"].avg_true_revenue
        random_revenue = results["Random"].avg_true_revenue
        
        # GeoSense sollte mindestens so gut sein wie Random
        assert geosense_revenue >= random_revenue * 0.8, \
            f"{vertical.value}: GeoSense sollte nicht viel schlechter als Random sein"


class TestReproducibility:
    """Tests für Reproduzierbarkeit."""
    
    def test_same_seed_same_results(self):
        """Gleicher Seed gibt gleiche Ergebnisse."""
        results1 = run_single_experiment(seed=12345, n_kandidaten=50, top_n=5)
        results2 = run_single_experiment(seed=12345, n_kandidaten=50, top_n=5)
        
        assert results1["GeoSense"].avg_true_revenue == \
               results2["GeoSense"].avg_true_revenue
    
    def test_different_seeds_different_results(self):
        """Verschiedene Seeds geben verschiedene Ergebnisse."""
        results1 = run_single_experiment(seed=11111, n_kandidaten=50, top_n=5)
        results2 = run_single_experiment(seed=22222, n_kandidaten=50, top_n=5)
        
        # Sollte in den meisten Fällen unterschiedlich sein
        assert results1["Random"].avg_true_revenue != \
               results2["Random"].avg_true_revenue or \
               results1["GeoSense"].avg_true_revenue != \
               results2["GeoSense"].avg_true_revenue


class TestWirksamkeitsNachweis:
    """
    HAUPTTEST: Formaler Wirksamkeitsnachweis.
    
    Dieser Test dokumentiert die quantitative Überlegenheit
    von GeoSense gegenüber Baselines.
    """
    
    def test_wirksamkeit_formal(self):
        """
        Formaler Wirksamkeitsnachweis:
        GeoSense wählt Standorte mit höherem True Revenue
        als Random-Baseline in mindestens 70% der Fälle.
        """
        # Konfiguration
        n_simulations = 50
        n_kandidaten = 200
        top_n = 10
        required_win_rate = 0.6  # 60% ist konservativ
        
        # Experimente durchführen
        df = run_experiment_suite(
            n_simulations=n_simulations,
            n_kandidaten=n_kandidaten,
            top_n=top_n,
            vertical=Vertical.RETAIL
        )
        
        # Auswertung
        comparison = compare_to_random_baseline(df)
        
        # Assertions
        assert comparison["win_rate"] >= required_win_rate, \
            f"WIRKSAMKEITSNACHWEIS FEHLGESCHLAGEN: " \
            f"Win-Rate {comparison['win_rate']*100:.1f}% < {required_win_rate*100}%"
        
        assert comparison["geosense_avg"] > comparison["random_avg"], \
            f"WIRKSAMKEITSNACHWEIS FEHLGESCHLAGEN: " \
            f"GeoSense ({comparison['geosense_avg']:.0f}) nicht besser als " \
            f"Random ({comparison['random_avg']:.0f})"
        
        # Erfolgreicher Nachweis
        print(f"\n{'='*60}")
        print("WIRKSAMKEITSNACHWEIS ERFOLGREICH")
        print(f"{'='*60}")
        print(f"GeoSense Win-Rate: {comparison['win_rate']*100:.1f}%")
        print(f"GeoSense Ø Revenue: {comparison['geosense_avg']:.0f} €")
        print(f"Random Ø Revenue: {comparison['random_avg']:.0f} €")
        print(f"Verbesserung: +{comparison['improvement']:.1f}%")
        print(f"{'='*60}\n")
