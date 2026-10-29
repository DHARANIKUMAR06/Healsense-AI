/**
 * HealthSense AI - Form submission and SHAP Bar Chart Renderer
 */

let shapChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
  const assessmentForm = document.getElementById('assessment-form');
  if (assessmentForm) {
    assessmentForm.addEventListener('submit', handleFormSubmit);
  }
});

async function handleFormSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const actionUrl = form.getAttribute('action') || window.location.pathname;
  
  const submitBtn = form.querySelector('.btn-submit');
  const btnText = submitBtn.querySelector('.btn-text');
  const spinner = submitBtn.querySelector('.spinner');

  // Disable button and show spinner
  submitBtn.disabled = true;
  if (btnText) btnText.style.display = 'none';
  if (spinner) spinner.style.display = 'inline-block';

  // Gather form data
  const formData = new FormData(form);
  const dataObj = {};
  formData.forEach((val, key) => {
    dataObj[key] = val;
  });

  try {
    const response = await fetch(actionUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(dataObj)
    });

    const result = await response.json();

    if (response.ok && result.status === 'success') {
      displayAssessmentResults(result);
    } else {
      alert('Error: ' + (result.message || 'Failed to complete assessment.'));
    }
  } catch (err) {
    console.error('Submission error:', err);
    alert('An unexpected network error occurred.');
  } finally {
    submitBtn.disabled = false;
    if (btnText) btnText.style.display = 'inline-block';
    if (spinner) spinner.style.display = 'none';
  }
}

function displayAssessmentResults(data) {
  const placeholderState = document.getElementById('placeholder-state');
  const resultContainer = document.getElementById('result-output-container');
  const badgeElement = document.getElementById('result-badge');
  const statusText = document.getElementById('result-status-text');
  const pctDisplay = document.getElementById('risk-pct-display');

  if (placeholderState) placeholderState.style.display = 'none';
  if (resultContainer) resultContainer.style.display = 'flex';

  const isPos = (data.result === 'Positive' || data.result.toLowerCase().includes('high'));

  if (badgeElement) {
    badgeElement.className = 'result-header-badge ' + (isPos ? 'positive' : 'negative');
  }

  if (statusText) {
    statusText.textContent = data.result + " Risk";
  }

  if (pctDisplay) {
    pctDisplay.textContent = data.risk_percentage + "%";
    pctDisplay.style.color = isPos ? '#e11d48' : '#16a34a';
  }

  // Render SHAP Horizontal Bar Chart
  if (data.shap_values && data.shap_values.length > 0) {
    renderShapChart(data.shap_values);
  }
}

function renderShapChart(shapData) {
  const ctx = document.getElementById('shapChartCanvas');
  if (!ctx) return;

  // Take top 7 features by absolute impact
  const sorted = shapData.slice(0, 7);
  const labels = sorted.map(item => item.feature);
  const values = sorted.map(item => item.impact_value);

  const backgroundColors = values.map(v => v >= 0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(34, 197, 94, 0.85)');
  const borderColors = values.map(v => v >= 0 ? '#dc2626' : '#16a34a');

  if (shapChartInstance) {
    shapChartInstance.destroy();
  }

  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'SHAP Feature Contribution',
        data: values,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const val = context.raw;
              const sign = val >= 0 ? '+' : '';
              return ` Impact: ${sign}${val} (Risk ${val >= 0 ? 'Increaser' : 'Decreaser'})`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(226, 232, 240, 0.8)' },
          title: {
            display: true,
            text: '← Decreases Risk  |  Increases Risk →',
            font: { size: 11, weight: 'bold' },
            color: '#64748b'
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            font: { size: 12, weight: '600' },
            color: '#1e293b'
          }
        }
      }
    }
  });
}
