const state = {
  hrRaw: [],
  hrClean: [],
  customerRaw: [],
  customerClean: [],
  hrReport: null,
  customerReport: null,
};

const palette = ['#5eead4', '#7c9cff', '#f59e0b', '#22c55e', '#f472b6', '#fb7185', '#38bdf8', '#a78bfa'];

function formatMoney(value) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value || 0);
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return Number(value).toFixed(digits);
}

function parseCsvText(text) {
  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results.data),
      error: reject,
    });
  });
}

async function loadCsvFromUrl(url) {
  const response = await fetch(url);
  const text = await response.text();
  return parseCsvText(text);
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

function median(values) {
  const valid = values.filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  if (!valid.length) return 0;
  const mid = Math.floor(valid.length / 2);
  return valid.length % 2 === 0 ? (valid[mid - 1] + valid[mid]) / 2 : valid[mid];
}

function quartile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function iqrBounds(values) {
  const sorted = values.filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  if (!sorted.length) return { lower: -Infinity, upper: Infinity };
  const q1 = quartile(sorted, 0.25);
  const q3 = quartile(sorted, 0.75);
  const iqr = q3 - q1;
  return { lower: q1 - 1.5 * iqr, upper: q3 + 1.5 * iqr };
}

function seededRandom(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function splitTrainTest(arr, testRatio = 0.2, seed = 42) {
  const rand = seededRandom(seed);
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  const testSize = Math.max(1, Math.floor(copy.length * testRatio));
  return { test: copy.slice(0, testSize), train: copy.slice(testSize) };
}

function mae(actual, predicted) {
  return actual.reduce((sum, value, i) => sum + Math.abs(value - predicted[i]), 0) / actual.length;
}

function mse(actual, predicted) {
  return actual.reduce((sum, value, i) => sum + (value - predicted[i]) ** 2, 0) / actual.length;
}

function rSquared(actual, predicted) {
  const mean = actual.reduce((a, b) => a + b, 0) / actual.length;
  const ssRes = actual.reduce((sum, value, i) => sum + (value - predicted[i]) ** 2, 0);
  const ssTot = actual.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

function fitSimpleLinearRegression(rows) {
  const xs = rows.map(r => r.Experience);
  const ys = rows.map(r => r.Salary);
  const xMean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const yMean = ys.reduce((a, b) => a + b, 0) / ys.length;
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < xs.length; i++) {
    numerator += (xs[i] - xMean) * (ys[i] - yMean);
    denominator += (xs[i] - xMean) ** 2;
  }
  const slope = denominator === 0 ? 0 : numerator / denominator;
  const intercept = yMean - slope * xMean;
  return {
    slope,
    intercept,
    predict: (x) => intercept + slope * x,
  };
}

function cleanHrData(rows) {
  const originalRows = rows.length;
  const normalized = rows.map(row => ({
    EmployeeID: row.EmployeeID ?? row.employeeid ?? '',
    Experience: Number.parseFloat(row.Experience ?? row.experience ?? ''),
    Rating: Number.parseFloat(row.Rating ?? row.rating ?? ''),
    Salary: Number.parseFloat(row.Salary ?? row.salary ?? ''),
    Department: (row.Department ?? row.department ?? 'Unknown').toString().trim() || 'Unknown',
  }));

  const deduped = [];
  const seen = new Set();
  for (const row of normalized) {
    const key = JSON.stringify(row);
    if (!seen.has(key)) {
      seen.add(key);
      deduped.push(row);
    }
  }

  const missingBefore = {
    Experience: deduped.filter(r => !Number.isFinite(r.Experience)).length,
    Rating: deduped.filter(r => !Number.isFinite(r.Rating)).length,
    Salary: deduped.filter(r => !Number.isFinite(r.Salary)).length,
  };

  const expMedian = median(deduped.map(r => r.Experience));
  const ratingMedian = median(deduped.map(r => r.Rating));
  const salaryMedian = median(deduped.map(r => r.Salary));

  const filled = deduped.map(r => ({
    ...r,
    Experience: Number.isFinite(r.Experience) ? r.Experience : expMedian,
    Rating: Number.isFinite(r.Rating) ? r.Rating : ratingMedian,
    Salary: Number.isFinite(r.Salary) ? r.Salary : salaryMedian,
  }));

  const realistic = filled.filter(r => r.Experience >= 0 && r.Experience <= 40 && r.Rating >= 1 && r.Rating <= 5 && r.Salary >= 20000 && r.Salary <= 250000);
  const expBounds = iqrBounds(realistic.map(r => r.Experience));
  const salaryBounds = iqrBounds(realistic.map(r => r.Salary));
  const cleaned = realistic
    .filter(r => r.Experience >= expBounds.lower && r.Experience <= expBounds.upper)
    .filter(r => r.Salary >= salaryBounds.lower && r.Salary <= salaryBounds.upper)
    .sort((a, b) => a.Experience - b.Experience);

  return {
    cleaned,
    summary: {
      originalRows,
      duplicatesRemoved: originalRows - deduped.length,
      medianFilled: missingBefore,
      unrealisticRemoved: filled.length - realistic.length,
      outliersRemoved: realistic.length - cleaned.length,
      finalRows: cleaned.length,
      medians: { expMedian, ratingMedian, salaryMedian },
    },
  };
}

function runSalaryAudit(rows, targetExperience = 15) {
  const { cleaned, summary } = cleanHrData(rows);
  const split = splitTrainTest(cleaned, 0.2, 42);
  const model = fitSimpleLinearRegression(split.train);
  const actual = split.test.map(r => r.Salary);
  const predicted = split.test.map(r => model.predict(r.Experience));
  const prediction = model.predict(targetExperience);
  return {
    cleaned,
    summary,
    model,
    metrics: {
      mae: mae(actual, predicted),
      mse: mse(actual, predicted),
      r2: rSquared(actual, predicted),
    },
    prediction,
  };
}

function euclidean(a, b) {
  return Math.sqrt(a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0));
}

function standardScale(rows, featureKeys) {
  const means = featureKeys.map(key => rows.reduce((sum, row) => sum + row[key], 0) / rows.length);
  const stds = featureKeys.map((key, idx) => {
    const variance = rows.reduce((sum, row) => sum + (row[key] - means[idx]) ** 2, 0) / rows.length;
    return Math.sqrt(variance) || 1;
  });
  const scaled = rows.map(row => featureKeys.map((key, idx) => (row[key] - means[idx]) / stds[idx]));
  return { scaled, means, stds };
}

function inverseScale(point, means, stds) {
  return point.map((value, i) => value * stds[i] + means[i]);
}

function kmeansPlusPlus(points, k, rand) {
  const centroids = [points[Math.floor(rand() * points.length)]];
  while (centroids.length < k) {
    const distances = points.map(point => {
      const minDist = Math.min(...centroids.map(c => euclidean(point, c) ** 2));
      return minDist;
    });
    const total = distances.reduce((a, b) => a + b, 0);
    let threshold = rand() * total;
    for (let i = 0; i < points.length; i++) {
      threshold -= distances[i];
      if (threshold <= 0) {
        centroids.push(points[i]);
        break;
      }
    }
    if (centroids.length < k && centroids.length === points.length) break;
  }
  while (centroids.length < k) centroids.push(points[Math.floor(rand() * points.length)]);
  return centroids.map(c => [...c]);
}

function runKMeans(points, k, seed = 42, maxIterations = 100) {
  const rand = seededRandom(seed);
  let centroids = kmeansPlusPlus(points, k, rand);
  let labels = new Array(points.length).fill(0);

  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;
    for (let i = 0; i < points.length; i++) {
      let bestLabel = 0;
      let bestDistance = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const dist = euclidean(points[i], centroids[c]);
        if (dist < bestDistance) {
          bestDistance = dist;
          bestLabel = c;
        }
      }
      if (labels[i] !== bestLabel) {
        labels[i] = bestLabel;
        changed = true;
      }
    }

    const newCentroids = Array.from({ length: k }, () => Array(points[0].length).fill(0));
    const counts = Array(k).fill(0);
    points.forEach((point, idx) => {
      const label = labels[idx];
      counts[label] += 1;
      point.forEach((value, dim) => {
        newCentroids[label][dim] += value;
      });
    });

    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        newCentroids[c] = [...points[Math.floor(rand() * points.length)]];
      } else {
        newCentroids[c] = newCentroids[c].map(v => v / counts[c]);
      }
    }

    const drift = centroids.reduce((sum, centroid, idx) => sum + euclidean(centroid, newCentroids[idx]), 0);
    centroids = newCentroids;
    if (!changed || drift < 1e-6) break;
  }

  const inertia = points.reduce((sum, point, idx) => sum + euclidean(point, centroids[labels[idx]]) ** 2, 0);
  return { centroids, labels, inertia };
}

function silhouetteScore(points, labels, k) {
  if (k <= 1) return 0;
  const clusters = Array.from({ length: k }, () => []);
  points.forEach((point, idx) => clusters[labels[idx]].push({ point, idx }));

  const scores = points.map((point, idx) => {
    const ownCluster = labels[idx];
    const ownPoints = clusters[ownCluster].filter(item => item.idx !== idx).map(item => item.point);
    const a = ownPoints.length ? ownPoints.reduce((sum, p) => sum + euclidean(point, p), 0) / ownPoints.length : 0;

    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ownCluster || clusters[c].length === 0) continue;
      const dist = clusters[c].reduce((sum, item) => sum + euclidean(point, item.point), 0) / clusters[c].length;
      if (dist < b) b = dist;
    }
    if (!Number.isFinite(b) && a === 0) return 0;
    return (b - a) / Math.max(a, b);
  });

  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

function detectElbow(ks, inertias) {
  if (ks.length <= 2) return ks[0];
  const x1 = ks[0], y1 = inertias[0];
  const x2 = ks[ks.length - 1], y2 = inertias[inertias.length - 1];
  let bestK = ks[0];
  let maxDistance = -1;
  for (let i = 0; i < ks.length; i++) {
    const x0 = ks[i], y0 = inertias[i];
    const distance = Math.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / Math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2);
    if (distance > maxDistance) {
      maxDistance = distance;
      bestK = ks[i];
    }
  }
  return bestK;
}

function cleanCustomerData(rows) {
  const originalRows = rows.length;
  const normalized = rows.map(row => ({
    CustomerID: row.CustomerID ?? row.customerid ?? '',
    Age: Number.parseFloat(row.Age ?? row.age ?? ''),
    MonthlySpend: Number.parseFloat(row.MonthlySpend ?? row.monthlyspend ?? ''),
    VisitsPerMonth: Number.parseFloat(row.VisitsPerMonth ?? row.visitspermonth ?? ''),
    LegacyLabel: (row.LegacyLabel ?? row.legacylabel ?? 'Unknown').toString().trim() || 'Unknown',
  }));

  const deduped = [];
  const seen = new Set();
  for (const row of normalized) {
    const key = JSON.stringify(row);
    if (!seen.has(key)) {
      seen.add(key);
      deduped.push(row);
    }
  }

  const missingBefore = {
    Age: deduped.filter(r => !Number.isFinite(r.Age)).length,
    MonthlySpend: deduped.filter(r => !Number.isFinite(r.MonthlySpend)).length,
    VisitsPerMonth: deduped.filter(r => !Number.isFinite(r.VisitsPerMonth)).length,
  };

  const ageMedian = median(deduped.map(r => r.Age));
  const spendMedian = median(deduped.map(r => r.MonthlySpend));
  const visitsMedian = median(deduped.map(r => r.VisitsPerMonth));

  const filled = deduped.map(r => ({
    ...r,
    Age: Number.isFinite(r.Age) ? r.Age : ageMedian,
    MonthlySpend: Number.isFinite(r.MonthlySpend) ? r.MonthlySpend : spendMedian,
    VisitsPerMonth: Number.isFinite(r.VisitsPerMonth) ? r.VisitsPerMonth : visitsMedian,
  }));

  const realistic = filled.filter(r => r.Age >= 18 && r.Age <= 75 && r.MonthlySpend >= 500 && r.MonthlySpend <= 20000 && r.VisitsPerMonth >= 1 && r.VisitsPerMonth <= 20);
  const ageBounds = iqrBounds(realistic.map(r => r.Age));
  const spendBounds = iqrBounds(realistic.map(r => r.MonthlySpend));
  const cleaned = realistic
    .filter(r => r.Age >= ageBounds.lower && r.Age <= ageBounds.upper)
    .filter(r => r.MonthlySpend >= spendBounds.lower && r.MonthlySpend <= spendBounds.upper)
    .sort((a, b) => a.MonthlySpend - b.MonthlySpend);

  return {
    cleaned,
    summary: {
      originalRows,
      duplicatesRemoved: originalRows - deduped.length,
      medianFilled: missingBefore,
      unrealisticRemoved: filled.length - realistic.length,
      outliersRemoved: realistic.length - cleaned.length,
      finalRows: cleaned.length,
    },
  };
}

function labelCluster(profile, maxSpend, minSpend) {
  if (profile.avgSpend >= maxSpend * 0.9) return 'VIP / High Roller';
  if (profile.avgAge <= 35 && profile.avgSpend <= minSpend * 1.25) return 'Young Value';
  if (profile.avgSpend >= (minSpend + maxSpend) / 2) return 'Premium Regular';
  return 'Family Standard';
}

function runCustomerSegmentation(rows, newCustomer) {
  const { cleaned, summary } = cleanCustomerData(rows);
  const featureKeys = ['Age', 'MonthlySpend'];
  const scaler = standardScale(cleaned, featureKeys);
  const ks = [2, 3, 4, 5, 6, 7, 8];
  const elbowRuns = ks.map(k => ({ k, ...runKMeans(scaler.scaled, k, 42 + k) }));
  const inertias = elbowRuns.map(run => run.inertia);
  const bestK = detectElbow(ks, inertias);
  const chosen = elbowRuns.find(run => run.k === bestK);
  const silhouette = silhouetteScore(scaler.scaled, chosen.labels, bestK);

  const profiles = Array.from({ length: bestK }, (_, clusterId) => {
    const members = cleaned.filter((_, idx) => chosen.labels[idx] === clusterId);
    const centroidOriginal = inverseScale(chosen.centroids[clusterId], scaler.means, scaler.stds);
    return {
      clusterId,
      size: members.length,
      avgAge: centroidOriginal[0],
      avgSpend: centroidOriginal[1],
      label: '',
    };
  });

  const spends = profiles.map(p => p.avgSpend);
  const maxSpend = Math.max(...spends);
  const minSpend = Math.min(...spends);
  profiles.forEach(profile => {
    profile.label = labelCluster(profile, maxSpend, minSpend);
  });

  const newScaled = featureKeys.map((key, idx) => ((newCustomer[key] - scaler.means[idx]) / scaler.stds[idx]));
  let predictedCluster = 0;
  let minDistance = Infinity;
  chosen.centroids.forEach((centroid, idx) => {
    const dist = euclidean(newScaled, centroid);
    if (dist < minDistance) {
      minDistance = dist;
      predictedCluster = idx;
    }
  });
  const matchedProfile = profiles.find(p => p.clusterId === predictedCluster);

  return {
    cleaned,
    summary,
    bestK,
    inertiaSeries: elbowRuns.map(run => ({ k: run.k, inertia: run.inertia })),
    labels: chosen.labels,
    centroids: chosen.centroids,
    profiles,
    silhouette,
    newCustomer: {
      ...newCustomer,
      clusterId: predictedCluster,
      label: matchedProfile?.label || 'Segment',
    },
  };
}

function renderMetricCards(targetId, items) {
  const container = document.getElementById(targetId);
  container.innerHTML = items.map(item => `
    <div class="metric-card">
      <span>${item.label}</span>
      <strong>${item.value}</strong>
    </div>
  `).join('');
}

function renderTable(targetId, rows, columns) {
  const container = document.getElementById(targetId);
  if (!rows.length) {
    container.innerHTML = '<div class="summary-box">No data available.</div>';
    return;
  }
  const header = columns.map(col => `<th>${col.label}</th>`).join('');
  const body = rows.map(row => `
    <tr>
      ${columns.map(col => `<td>${col.render ? col.render(row[col.key], row) : row[col.key]}</td>`).join('')}
    </tr>
  `).join('');
  container.innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderHr(report, targetExperience) {
  renderMetricCards('hrMetrics', [
    { label: 'Rows after cleaning', value: report.summary.finalRows },
    { label: 'MAE', value: formatMoney(report.metrics.mae) },
    { label: 'MSE', value: formatNumber(report.metrics.mse, 0) },
    { label: 'R-Squared', value: formatNumber(report.metrics.r2, 3) },
  ]);

  const lineXs = [];
  const minX = Math.min(...report.cleaned.map(r => r.Experience));
  const maxX = Math.max(...report.cleaned.map(r => r.Experience));
  for (let x = minX; x <= maxX; x += 0.5) lineXs.push(x);
  const lineYs = lineXs.map(x => report.model.predict(x));

  Plotly.newPlot('salaryChart', [
    {
      x: report.cleaned.map(r => r.Experience),
      y: report.cleaned.map(r => r.Salary),
      mode: 'markers',
      type: 'scatter',
      name: 'Employees',
      marker: { color: '#5eead4', size: 10, opacity: 0.8 },
      hovertemplate: 'Experience: %{x} years<br>Salary: %{y:$,.0f}<extra></extra>',
    },
    {
      x: lineXs,
      y: lineYs,
      mode: 'lines',
      type: 'scatter',
      name: 'Regression line',
      line: { color: '#f59e0b', width: 3 },
      hovertemplate: 'Experience: %{x} years<br>Predicted Salary: %{y:$,.0f}<extra></extra>',
    },
  ], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e9f2ff', family: 'Inter, sans-serif' },
    margin: { l: 55, r: 20, t: 10, b: 55 },
    xaxis: { title: 'Years of Experience', gridcolor: 'rgba(255,255,255,0.08)' },
    yaxis: { title: 'Salary', gridcolor: 'rgba(255,255,255,0.08)' },
    legend: { orientation: 'h', y: 1.12 },
  }, { responsive: true, displayModeBar: false });

  document.getElementById('hrQualitySummary').innerHTML = `
    <ul>
      <li><strong>${report.summary.originalRows}</strong> raw HR rows loaded.</li>
      <li><strong>${report.summary.duplicatesRemoved}</strong> duplicate records removed.</li>
      <li>Median fill applied for missing values: Experience <strong>${report.summary.medianFilled.Experience}</strong>, Rating <strong>${report.summary.medianFilled.Rating}</strong>, Salary <strong>${report.summary.medianFilled.Salary}</strong>.</li>
      <li><strong>${report.summary.unrealisticRemoved}</strong> unrealistic records excluded using business rules.</li>
      <li><strong>${report.summary.outliersRemoved}</strong> outliers removed with the IQR method.</li>
      <li>Regression equation: <code class="inline">Salary = ${formatNumber(report.model.slope, 2)} × Experience + ${formatNumber(report.model.intercept, 2)}</code></li>
      <li>The live prediction for <strong>${targetExperience} years</strong> is <strong>${formatMoney(report.prediction)}</strong>.</li>
    </ul>
  `;

  document.getElementById('salaryPredictionBanner').innerHTML = `
    <strong>Salary prediction ready:</strong> Based on the cleaned HR data, the estimated salary for a candidate with <strong>${targetExperience} years of experience</strong> is <strong>${formatMoney(report.prediction)}</strong>.
  `;

  renderTable('hrTable', report.cleaned.slice(0, 18), [
    { key: 'EmployeeID', label: 'Employee ID' },
    { key: 'Experience', label: 'Experience (Years)' },
    { key: 'Rating', label: 'Rating' },
    { key: 'Salary', label: 'Salary', render: value => formatMoney(value) },
    { key: 'Department', label: 'Department' },
  ]);
  document.getElementById('hrTableCount').textContent = `${report.cleaned.length} rows after cleaning`;
  document.getElementById('heroHrRows').textContent = report.summary.finalRows;
  document.getElementById('heroSalaryPrediction').textContent = formatMoney(report.prediction);
}

function renderCustomer(report) {
  renderMetricCards('customerMetrics', [
    { label: 'Rows after cleaning', value: report.summary.finalRows },
    { label: 'Optimal K', value: report.bestK },
    { label: 'Silhouette Score', value: formatNumber(report.silhouette, 3) },
    { label: 'New Customer Segment', value: report.newCustomer.label },
  ]);

  Plotly.newPlot('elbowChart', [{
    x: report.inertiaSeries.map(item => item.k),
    y: report.inertiaSeries.map(item => item.inertia),
    mode: 'lines+markers',
    type: 'scatter',
    line: { color: '#7c9cff', width: 3 },
    marker: { size: 10, color: '#5eead4' },
    hovertemplate: 'K = %{x}<br>Inertia = %{y:.2f}<extra></extra>',
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e9f2ff', family: 'Inter, sans-serif' },
    margin: { l: 55, r: 20, t: 10, b: 55 },
    xaxis: { title: 'Number of clusters (K)', dtick: 1, gridcolor: 'rgba(255,255,255,0.08)' },
    yaxis: { title: 'Inertia', gridcolor: 'rgba(255,255,255,0.08)' },
  }, { responsive: true, displayModeBar: false });

  const clusterTraces = report.profiles.map(profile => {
    const members = report.cleaned.filter((_, idx) => report.labels[idx] === profile.clusterId);
    return {
      x: members.map(r => r.Age),
      y: members.map(r => r.MonthlySpend),
      mode: 'markers',
      type: 'scatter',
      name: profile.label,
      marker: { size: 10, color: palette[profile.clusterId % palette.length], opacity: 0.85 },
      hovertemplate: `${profile.label}<br>Age: %{x}<br>Spend: %{y:$,.0f}<extra></extra>`,
    };
  });

  clusterTraces.push({
    x: [report.newCustomer.Age],
    y: [report.newCustomer.MonthlySpend],
    mode: 'markers',
    type: 'scatter',
    name: 'New customer',
    marker: { size: 16, color: '#ffffff', line: { color: '#f59e0b', width: 3 }, symbol: 'diamond' },
    hovertemplate: 'New customer<br>Age: %{x}<br>Spend: %{y:$,.0f}<extra></extra>',
  });

  Plotly.newPlot('clusterChart', clusterTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e9f2ff', family: 'Inter, sans-serif' },
    margin: { l: 55, r: 20, t: 10, b: 55 },
    xaxis: { title: 'Age', gridcolor: 'rgba(255,255,255,0.08)' },
    yaxis: { title: 'Monthly Spend', gridcolor: 'rgba(255,255,255,0.08)' },
    legend: { orientation: 'h', y: 1.14 },
  }, { responsive: true, displayModeBar: false });

  document.getElementById('customerClassificationBanner').innerHTML = `
    <strong>Segmentation ready:</strong> The dashboard selected <strong>${report.bestK}</strong> customer groups. A new customer aged <strong>${report.newCustomer.Age}</strong> with monthly spend of <strong>${formatMoney(report.newCustomer.MonthlySpend)}</strong> belongs to <strong>${report.newCustomer.label}</strong>.
  `;

  renderTable('clusterTable', report.profiles.map(profile => ({
    Cluster: profile.clusterId,
    Segment: profile.label,
    Customers: profile.size,
    AverageAge: profile.avgAge,
    AverageSpend: profile.avgSpend,
  })), [
    { key: 'Cluster', label: 'Cluster' },
    { key: 'Segment', label: 'Business Label' },
    { key: 'Customers', label: 'Customers' },
    { key: 'AverageAge', label: 'Average Age', render: value => formatNumber(value, 1) },
    { key: 'AverageSpend', label: 'Average Monthly Spend', render: value => formatMoney(value) },
  ]);
  document.getElementById('clusterTableCount').textContent = `${report.profiles.length} clusters`;
  document.getElementById('heroCustomerRows').textContent = report.summary.finalRows;
  document.getElementById('heroBestK').textContent = report.bestK;
}

async function loadHrSample() {
  state.hrRaw = await loadCsvFromUrl('hr_dirty_dataset.csv');
  document.getElementById('runHrBtn').click();
}

async function loadCustomerSample() {
  state.customerRaw = await loadCsvFromUrl('customer_segmentation_data.csv');
  document.getElementById('runCustomerBtn').click();
}

function setupEvents() {
  document.getElementById('runHrBtn').addEventListener('click', () => {
    if (!state.hrRaw.length) return;
    const experience = Number(document.getElementById('experienceInput').value) || 15;
    state.hrReport = runSalaryAudit(state.hrRaw, experience);
    state.hrClean = state.hrReport.cleaned;
    renderHr(state.hrReport, experience);
  });

  document.getElementById('runCustomerBtn').addEventListener('click', () => {
    if (!state.customerRaw.length) return;
    const age = Number(document.getElementById('ageInput').value) || 25;
    const spend = Number(document.getElementById('spendInput').value) || 2500;
    state.customerReport = runCustomerSegmentation(state.customerRaw, { Age: age, MonthlySpend: spend });
    state.customerClean = state.customerReport.cleaned;
    renderCustomer(state.customerReport);
  });

  document.getElementById('loadHrSampleBtn').addEventListener('click', loadHrSample);
  document.getElementById('loadCustomerSampleBtn').addEventListener('click', loadCustomerSample);
  document.getElementById('loadAllSamplesBtn').addEventListener('click', async () => {
    await Promise.all([loadHrSample(), loadCustomerSample()]);
  });

  document.getElementById('hrFileInput').addEventListener('change', async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await readFileAsText(file);
    state.hrRaw = await parseCsvText(text);
    document.getElementById('runHrBtn').click();
  });

  document.getElementById('customerFileInput').addEventListener('change', async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await readFileAsText(file);
    state.customerRaw = await parseCsvText(text);
    document.getElementById('runCustomerBtn').click();
  });
}

setupEvents();
loadHrSample();
loadCustomerSample();
