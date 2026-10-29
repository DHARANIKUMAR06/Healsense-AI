/**
 * HealthSense AI - Dashboard Analytics & Chart.js Visualizer
 */

let barChartInstance = null;
let donutChartInstance = null;
let lineChartInstance = null;
let pollTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  fetchDashboardStats();
  // Auto-refresh stats every 30 seconds
  pollTimer = setInterval(fetchDashboardStats, 30000);
});

async function fetchDashboardStats() {
  try {
    const res = await fetch('/api/dashboard-stats');
    const data = await res.json();

    if (data.is_empty) {
      renderEmptyState();
    } else {
      renderDashboardData(data);
    }
  } catch (err) {
    console.error('Failed to fetch dashboard stats:', err);
  }
}

function renderEmptyState() {
  const emptyState = document.getElementById('dash-empty-state');
  const dashContent = document.getElementById('dash-full-content');

  if (emptyState) emptyState.style.display = 'block';
  if (dashContent) dashContent.style.display = 'none';
}

function renderDashboardData(data) {
  const emptyState = document.getElementById('dash-empty-state');
  const dashContent = document.getElementById('dash-full-content');

  if (emptyState) emptyState.style.display = 'none';
  if (dashContent) dashContent.style.display = 'block';

  // 1. Render Summary Cards
  renderSummaryCards(data.summary_cards);

  // 2. Render Charts
  renderBarChart(data.assessments_by_disease);
  renderDonutChart(data.risk_distribution);
  renderLineChart(data.assessments_over_time);

  // 3. Render Recent Activity Table
  renderRecentTable(data.recent_activity);
}

function renderSummaryCards(cards) {
  const container = document.getElementById('summary-cards-container');
  if (!container) return;

  const icons = {
    'diabetes': 'fa-syringe',
    'heart': 'fa-heart-crack',
    'kidney': 'fa-filter',
    'malaria': 'fa-mosquito',
    'dengue': 'fa-virus'
  };

  container.innerHTML = cards.map(c => {
    let riskClass = 'risk-low';
    if (c.positive_percentage > 50) riskClass = 'risk-high';
    else if (c.positive_percentage > 25) riskClass = 'risk-med';

    let trendIcon = '<span class="trend-indicator"><i class="fa-solid fa-minus"></i> Stable</span>';
    if (c.trend === 'up') trendIcon = '<span class="trend-indicator up"><i class="fa-solid fa-arrow-up"></i> Up</span>';
    else if (c.trend === 'down') trendIcon = '<span class="trend-indicator down"><i class="fa-solid fa-arrow-down"></i> Down</span>';

    return `
      <div class="summary-card ${riskClass}">
        <div class="summary-card-header">
          <div class="summary-card-title">${c.disease}</div>
          <div class="summary-card-icon">
            <i class="fa-solid ${icons[c.disease_key] || 'fa-notes-medical'}"></i>
          </div>
        </div>
        <div class="summary-card-count">${c.total_assessments}</div>
        <div class="summary-card-footer">
          <span>${c.positive_percentage}% Positive</span>
          ${trendIcon}
        </div>
      </div>
    `;
  }).join('');
}

function renderBarChart(barData) {
  const ctx = document.getElementById('barChartCanvas');
  if (!ctx) return;

  if (barChartInstance) barChartInstance.destroy();

  barChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: barData.labels,
      datasets: [{
        label: 'Assessments',
        data: barData.counts,
        backgroundColor: [
          'rgba(37, 99, 235, 0.85)',
          'rgba(244, 63, 94, 0.85)',
          'rgba(6, 182, 212, 0.85)',
          'rgba(245, 158, 11, 0.85)',
          'rgba(16, 185, 129, 0.85)'
        ],
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
        x: { grid: { display: false } }
      }
    }
  });
}

function renderDonutChart(distData) {
  const ctx = document.getElementById('donutChartCanvas');
  if (!ctx) return;

  if (donutChartInstance) donutChartInstance.destroy();

  donutChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Positive Risk', 'Negative Risk'],
      datasets: [{
        data: [distData.positive_pct, distData.negative_pct],
        backgroundColor: ['#f43f5e', '#10b981'],
        borderWidth: 2,
        borderColor: '#ffffff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { font: { size: 12, weight: '600' } } }
      },
      cutout: '68%'
    }
  });
}

function renderLineChart(overTimeData) {
  const ctx = document.getElementById('lineChartCanvas');
  if (!ctx) return;

  if (lineChartInstance) lineChartInstance.destroy();

  const colors = {
    'diabetes': '#2563eb',
    'heart': '#f43f5e',
    'kidney': '#06b6d4',
    'malaria': '#f59e0b',
    'dengue': '#10b981'
  };

  const labelsMap = {
    'diabetes': 'Diabetes',
    'heart': 'Heart Disease',
    'kidney': 'Kidney Disease',
    'malaria': 'Malaria',
    'dengue': 'Dengue'
  };

  const datasets = Object.keys(overTimeData.series).map(d => {
    return {
      label: labelsMap[d] || d,
      data: overTimeData.series[d],
      borderColor: colors[d] || '#64748b',
      tension: 0.3,
      fill: false,
      borderWidth: 2,
      pointRadius: 3
    };
  });

  lineChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: overTimeData.dates,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11, weight: '600' } } }
      },
      scales: {
        y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
        x: { grid: { display: false } }
      }
    }
  });
}

function renderRecentTable(rows) {
  const tbody = document.getElementById('recent-table-body');
  if (!tbody) return;

  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: #94a3b8;">No recent activity records.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const isPos = r.result === 'Positive' || r.result.includes('High') || r.result === 'Tachycardia';
    const badgeClass = isPos ? 'pos' : 'neg';
    const riskStr = r.risk_percentage !== null ? `${r.risk_percentage}%` : 'N/A';

    return `
      <tr>
        <td style="font-size: 12.5px; color: #64748b;">${r.timestamp}</td>
        <td style="font-weight: 600; color: #1e293b;">${r.disease_type}</td>
        <td><span class="badge-result ${badgeClass}">${r.result}</span></td>
        <td style="font-weight: 700; color: ${isPos ? '#e11d48' : '#16a34a'};">${riskStr}</td>
      </tr>
    `;
  }).join('');
}
