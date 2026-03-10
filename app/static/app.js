/**
 * GeoSense Standort-Assistent - Client-side JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initTabs();
    initPresets();
    initFormValidation();
    initSummary();
});

/**
 * Tab Navigation
 */
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabId = this.dataset.tab;
            
            // Deactivate all tabs
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Activate clicked tab
            this.classList.add('active');
            this.setAttribute('aria-selected', 'true');
            const tabContent = document.getElementById('tab-' + tabId);
            if (tabContent) {
                tabContent.classList.add('active');
            }
        });

        btn.setAttribute('role', 'tab');
        btn.setAttribute('aria-selected', btn.classList.contains('active') ? 'true' : 'false');
    });
}

/**
 * Preset Buttons
 */
function initPresets() {
    const presetButtons = document.querySelectorAll('.preset-btn');
    
    presetButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const presetName = this.dataset.preset;
            
            // Update active state
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // Update hidden input
            const presetInput = document.getElementById('preset');
            if (presetInput) {
                presetInput.value = presetName;
            }
            
            // Apply preset values
            applyPreset(presetName);
        });
    });
}

/**
 * Apply preset configuration
 */
function applyPreset(presetName) {
    const presets = {
        balanced: {
            optimierungsziel: 'roi_max',
            risikoappetit: 'neutral',
            gewicht_fussgaenger: 0.3,
            gewicht_pendler: 0.3,
            gewicht_drive_by: 0.2,
            gewicht_kaufkraft: 0.4,
            gewicht_kannibalisierung: 0.3
        },
        growth: {
            optimierungsziel: 'umsatz_max',
            risikoappetit: 'aggressiv',
            gewicht_fussgaenger: 0.4,
            gewicht_pendler: 0.35,
            gewicht_drive_by: 0.25,
            gewicht_kaufkraft: 0.5,
            gewicht_kannibalisierung: 0.15
        },
        low_risk: {
            optimierungsziel: 'risiko_min',
            risikoappetit: 'konservativ',
            gewicht_fussgaenger: 0.25,
            gewicht_pendler: 0.25,
            gewicht_drive_by: 0.15,
            gewicht_kaufkraft: 0.35,
            gewicht_kannibalisierung: 0.5
        }
    };

    const preset = presets[presetName];
    if (!preset) return;

    // Apply each value
    for (const [key, value] of Object.entries(preset)) {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
            
            // Trigger input event for range sliders to update displayed value
            element.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
    
    updateSummary();
}

/**
 * Form Validation
 */
function initFormValidation() {
    const form = document.getElementById('analyze-form');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            const submitBtn = document.getElementById('submit-btn');
            
            // Disable button and show loading state
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner"></span> Analyse läuft...';
            }
            
            // Basic validation
            const stadtPlz = document.getElementById('stadt_plz');
            if (stadtPlz && !stadtPlz.value.trim()) {
                e.preventDefault();
                alert('Bitte geben Sie eine Stadt oder PLZ ein.');
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = 'Standortanalyse starten';
                }
                return false;
            }
        });
    }
}

/**
 * Update Summary
 */
function initSummary() {
    // Update summary on key field changes
    const keyFields = ['vertical', 'stadt_plz', 'radius_km', 'store_format', 'optimierungsziel', 'preset'];
    
    keyFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('change', updateSummary);
        }
    });
    
    // Initial summary update
    updateSummary();
}

function updateSummary() {
    const summaryContent = document.getElementById('summary-content');
    if (!summaryContent) return;
    
    const vertical = document.getElementById('vertical');
    const stadt = document.getElementById('stadt_plz');
    const radius = document.getElementById('radius_km');
    const format = document.getElementById('store_format');
    const ziel = document.getElementById('optimierungsziel');
    
    const verticalNames = {
        'tankstelle': 'Tankstelle',
        'retail': 'Retail/Einzelhandel',
        'gastro': 'Gastronomie',
        'fitness': 'Fitnessstudio',
        'drogerie': 'Drogerie',
        'baeckerei': 'Bäckerei'
    };
    
    const zielNames = {
        'umsatz_max': 'Umsatz maximieren',
        'roi_max': 'ROI maximieren',
        'risiko_min': 'Risiko minimieren',
        'payback_min': 'Payback minimieren'
    };
    
    let html = '<ul style="list-style: none; padding: 0; margin: 0;">';
    
    if (vertical) {
        html += `<li><strong>Branche:</strong> ${verticalNames[vertical.value] || vertical.value}</li>`;
    }
    if (stadt && radius) {
        html += `<li><strong>Suchgebiet:</strong> ${stadt.value}, ${radius.value} km Radius</li>`;
    }
    if (format) {
        html += `<li><strong>Store-Format:</strong> ${format.value}</li>`;
    }
    if (ziel) {
        html += `<li><strong>Optimierungsziel:</strong> ${zielNames[ziel.value] || ziel.value}</li>`;
    }
    
    html += '</ul>';
    
    summaryContent.innerHTML = html;
}

/**
 * Toggle Details Row
 */
function toggleDetails(id) {
    const row = document.getElementById('details-' + id);
    const btn = document.querySelector(`[data-standort-id="${id}"] .expand-btn`);
    
    if (!row) return;
    const icon = '<svg class="expand-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>';

    if (row.style.display === 'none' || row.style.display === '') {
        row.style.display = 'table-row';
        if (btn) {
            btn.classList.add('active');
            btn.setAttribute('aria-expanded', 'true');
            btn.innerHTML = `${icon} Schließen`;
        }
    } else {
        row.style.display = 'none';
        if (btn) {
            btn.classList.remove('active');
            btn.setAttribute('aria-expanded', 'false');
            btn.innerHTML = `${icon} Details`;
        }
    }
}

/**
 * Format number as currency
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('de-DE', {
        style: 'currency',
        currency: 'EUR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Format number as percentage
 */
function formatPercent(value, decimals = 1) {
    return value.toFixed(decimals) + '%';
}
