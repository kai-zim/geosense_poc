"""
Tests für die Pydantic-Modelle und Input-Validierung.
"""
import pytest
from pydantic import ValidationError

from app.models import (
    AnalyzeInput, GrunddatenInput, GeschaeftsmodellInput,
    NachfrageMobilitaetInput, KaufkraftDemografieInput,
    WettbewerbInput, StandortInfrastrukturInput, WetterSaisonInput,
    ZieleRisikoInput, ModellOptionen, Vertical, StoreFormat,
    Oeffnungszeiten, Optimierungsziel, Risikoappetit, Preset
)


class TestGrunddatenValidation:
    """Tests für GrunddatenInput-Validierung."""
    
    def test_default_values(self):
        """Standardwerte werden korrekt gesetzt."""
        input = GrunddatenInput()
        assert input.vertical == Vertical.TANKSTELLE
        assert input.stadt_plz == "Berlin"
        assert input.radius_km == 10.0
        assert input.kandidaten_anzahl == 200
        assert input.top_n == 10
        
    def test_valid_radius(self):
        """Gültiger Radius wird akzeptiert."""
        input = GrunddatenInput(radius_km=25.0)
        assert input.radius_km == 25.0
        
    def test_radius_too_small(self):
        """Zu kleiner Radius wird abgelehnt."""
        with pytest.raises(ValidationError):
            GrunddatenInput(radius_km=0.5)
            
    def test_radius_too_large(self):
        """Zu großer Radius wird abgelehnt."""
        with pytest.raises(ValidationError):
            GrunddatenInput(radius_km=100)
            
    def test_kandidaten_anzahl_range(self):
        """Kandidaten-Anzahl muss im gültigen Bereich liegen."""
        # Gültig
        input = GrunddatenInput(kandidaten_anzahl=500)
        assert input.kandidaten_anzahl == 500
        
        # Zu klein
        with pytest.raises(ValidationError):
            GrunddatenInput(kandidaten_anzahl=5)
            
        # Zu groß
        with pytest.raises(ValidationError):
            GrunddatenInput(kandidaten_anzahl=5000)
            
    def test_top_n_range(self):
        """Top-N muss im gültigen Bereich liegen."""
        # Gültig
        input = GrunddatenInput(top_n=20)
        assert input.top_n == 20
        
        # Zu klein
        with pytest.raises(ValidationError):
            GrunddatenInput(top_n=1)
            
    def test_stadt_plz_length(self):
        """Stadt/PLZ muss Mindestlänge haben."""
        with pytest.raises(ValidationError):
            GrunddatenInput(stadt_plz="X")
            
    def test_optional_seed(self):
        """Seed ist optional."""
        input1 = GrunddatenInput()
        assert input1.seed is None
        
        input2 = GrunddatenInput(seed=42)
        assert input2.seed == 42


class TestGeschaeftsmodellValidation:
    """Tests für GeschaeftsmodellInput-Validierung."""
    
    def test_default_values(self):
        """Standardwerte werden korrekt gesetzt."""
        input = GeschaeftsmodellInput()
        assert input.store_format == StoreFormat.STANDARD
        assert input.flaeche_qm == 150.0
        
    def test_flaeche_range(self):
        """Fläche muss im gültigen Bereich liegen."""
        # Gültig
        input = GeschaeftsmodellInput(flaeche_qm=500)
        assert input.flaeche_qm == 500
        
        # Zu klein
        with pytest.raises(ValidationError):
            GeschaeftsmodellInput(flaeche_qm=10)
            
    def test_miete_range(self):
        """Miete muss im gültigen Bereich liegen."""
        # Gültig
        input = GeschaeftsmodellInput(miete_pro_qm=50)
        assert input.miete_pro_qm == 50
        
        # Zu klein
        with pytest.raises(ValidationError):
            GeschaeftsmodellInput(miete_pro_qm=1)
            
    def test_capex_range(self):
        """CAPEX muss im gültigen Bereich liegen."""
        input = GeschaeftsmodellInput(capex_euro=500000)
        assert input.capex_euro == 500000
        
        with pytest.raises(ValidationError):
            GeschaeftsmodellInput(capex_euro=1000)
            
    def test_zielkunden_list(self):
        """Zielkunden-Liste funktioniert."""
        input = GeschaeftsmodellInput(zielkunden=["pendler", "touristen"])
        assert len(input.zielkunden) == 2
        assert "pendler" in input.zielkunden


class TestNachfrageValidation:
    """Tests für NachfrageMobilitaetInput-Validierung."""
    
    def test_gewichte_range(self):
        """Gewichte müssen zwischen 0 und 1 liegen."""
        # Gültig
        input = NachfrageMobilitaetInput(gewicht_fussgaenger=0.5)
        assert input.gewicht_fussgaenger == 0.5
        
        # Zu klein
        with pytest.raises(ValidationError):
            NachfrageMobilitaetInput(gewicht_fussgaenger=-0.1)
            
        # Zu groß
        with pytest.raises(ValidationError):
            NachfrageMobilitaetInput(gewicht_fussgaenger=1.5)


class TestKaufkraftValidation:
    """Tests für KaufkraftDemografieInput-Validierung."""
    
    def test_altersgruppen_range(self):
        """Altersgruppen-Gewichte müssen zwischen 0 und 1 liegen."""
        input = KaufkraftDemografieInput(
            altersgruppe_18_25=0.25,
            altersgruppe_26_40=0.35,
            altersgruppe_41_65=0.25,
            altersgruppe_65_plus=0.15
        )
        assert input.altersgruppe_18_25 == 0.25


class TestWettbewerbValidation:
    """Tests für WettbewerbInput-Validierung."""
    
    def test_konkurrenz_radius_range(self):
        """Konkurrenzradius muss im gültigen Bereich liegen."""
        input = WettbewerbInput(konkurrenz_radius_km=5.0)
        assert input.konkurrenz_radius_km == 5.0
        
        with pytest.raises(ValidationError):
            WettbewerbInput(konkurrenz_radius_km=0.1)


class TestInfrastrukturValidation:
    """Tests für StandortInfrastrukturInput-Validierung."""
    
    def test_parkplaetze_range(self):
        """Parkplätze müssen >= 0 sein."""
        input = StandortInfrastrukturInput(parkplaetze_min=0)
        assert input.parkplaetze_min == 0
        
    def test_oepnv_range(self):
        """ÖPNV-Entfernung muss im gültigen Bereich liegen."""
        input = StandortInfrastrukturInput(oepnv_naehe_max_min=15)
        assert input.oepnv_naehe_max_min == 15
        
        with pytest.raises(ValidationError):
            StandortInfrastrukturInput(oepnv_naehe_max_min=0)


class TestZieleValidation:
    """Tests für ZieleRisikoInput-Validierung."""
    
    def test_mindest_roi_range(self):
        """Mindest-ROI muss im gültigen Bereich liegen."""
        input = ZieleRisikoInput(mindest_roi_prozent=20)
        assert input.mindest_roi_prozent == 20
        
        with pytest.raises(ValidationError):
            ZieleRisikoInput(mindest_roi_prozent=-5)
            
    def test_payback_range(self):
        """Max-Payback muss im gültigen Bereich liegen."""
        input = ZieleRisikoInput(max_payback_monate=48)
        assert input.max_payback_monate == 48
        
        with pytest.raises(ValidationError):
            ZieleRisikoInput(max_payback_monate=3)


class TestAnalyzeInputIntegration:
    """Integrationstests für komplette AnalyzeInput."""
    
    def test_full_input_creation(self):
        """Kompletter Input kann erstellt werden."""
        input = AnalyzeInput(
            grunddaten=GrunddatenInput(
                vertical=Vertical.RETAIL,
                stadt_plz="München",
                radius_km=15
            ),
            geschaeftsmodell=GeschaeftsmodellInput(
                store_format=StoreFormat.FLAGSHIP,
                flaeche_qm=500
            ),
            ziele=ZieleRisikoInput(
                optimierungsziel=Optimierungsziel.UMSATZ_MAX,
                risikoappetit=Risikoappetit.AGGRESSIV
            )
        )
        
        assert input.grunddaten.vertical == Vertical.RETAIL
        assert input.geschaeftsmodell.store_format == StoreFormat.FLAGSHIP
        assert input.ziele.optimierungsziel == Optimierungsziel.UMSATZ_MAX
        
    def test_default_nested_objects(self):
        """Verschachtelte Objekte haben Standardwerte."""
        input = AnalyzeInput()
        
        assert input.grunddaten is not None
        assert input.geschaeftsmodell is not None
        assert input.nachfrage is not None
        assert input.kaufkraft is not None
        assert input.wettbewerb is not None
        assert input.infrastruktur is not None
        assert input.wetter is not None
        assert input.ziele is not None
        assert input.optionen is not None


class TestEnumValues:
    """Tests für Enum-Werte."""
    
    def test_vertical_values(self):
        """Alle Vertikal-Werte sind gültig."""
        for v in Vertical:
            assert v.value in [
                "tankstelle", "retail", "gastro", 
                "fitness", "drogerie", "baeckerei"
            ]
            
    def test_store_format_values(self):
        """Alle Store-Format-Werte sind gültig."""
        for f in StoreFormat:
            assert f.value in ["klein", "standard", "flagship"]
            
    def test_optimierungsziel_values(self):
        """Alle Optimierungsziel-Werte sind gültig."""
        for o in Optimierungsziel:
            assert o.value in [
                "umsatz_max", "roi_max", 
                "risiko_min", "payback_min"
            ]
            
    def test_preset_values(self):
        """Alle Preset-Werte sind gültig."""
        for p in Preset:
            assert p.value in ["balanced", "growth", "low_risk", "custom"]
