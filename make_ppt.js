"use strict";
const pptxgen = require("pptxgenjs");

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  navy:    "021B4E",
  blue:    "065A82",
  teal:    "1C7293",
  accent:  "00B4D8",
  white:   "FFFFFF",
  offwhite:"F0F4F8",
  light:   "E8F4FD",
  muted:   "5C7A99",
  dark:    "021B4E",
  low:     "27AE60",
  med:     "F39C12",
  high:    "E74C3C",
  card:    "FFFFFF",
};

const FONT_H = "Georgia";
const FONT_B = "Calibri";

// ── Helpers ───────────────────────────────────────────────────────────────────
function darkSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: C.navy };
  return s;
}
function lightSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: C.offwhite };
  return s;
}

// Accent bar on left edge of a card
function accentBar(s, x, y, h, color) {
  s.addShape("rect", { x, y, w: 0.07, h, fill: { color }, line: { color, width: 0 } });
}

// Card with optional left accent
function card(s, x, y, w, h, fillColor, accentColor) {
  s.addShape("rect", {
    x, y, w, h,
    fill: { color: fillColor || C.card },
    line: { color: "D0DDE8", width: 1 },
    shadow: { type: "outer", color: "000000", opacity: 0.08, blur: 6, offset: 3, angle: 135 },
  });
  if (accentColor) accentBar(s, x, y, h, accentColor);
}

// Section header bar (dark)
function sectionBar(s, title) {
  s.addShape("rect", { x: 0, y: 0, w: 10, h: 0.72, fill: { color: C.navy }, line: { color: C.navy, width: 0 } });
  s.addText(title, {
    x: 0.45, y: 0, w: 9.1, h: 0.72, margin: 0,
    fontSize: 18, fontFace: FONT_H, bold: true, color: C.white, valign: "middle",
  });
  // accent dot
  s.addShape("rect", { x: 0, y: 0, w: 0.07, h: 0.72, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
}

// Footer
function footer(s, light) {
  const bg = light ? C.blue : "0A2540";
  s.addShape("rect", { x: 0, y: 5.35, w: 10, h: 0.275, fill: { color: bg }, line: { color: bg, width: 0 } });
  s.addText("Chennai Groundwater AI  |  AI-Powered Prediction & Risk Assessment", {
    x: 0.3, y: 5.35, w: 9.4, h: 0.275, margin: 0,
    fontSize: 8, fontFace: FONT_B, color: "AACCE0", valign: "middle",
  });
}

// Stat callout box
function statBox(s, x, y, value, label, accent) {
  card(s, x, y, 2.1, 1.1, C.card, accent || C.accent);
  s.addText(value, {
    x: x + 0.12, y: y + 0.05, w: 1.9, h: 0.6, margin: 0,
    fontSize: 30, fontFace: FONT_H, bold: true, color: accent || C.blue, align: "center",
  });
  s.addText(label, {
    x: x + 0.12, y: y + 0.62, w: 1.9, h: 0.38, margin: 0,
    fontSize: 10, fontFace: FONT_B, color: C.muted, align: "center",
  });
}

// Bullet list helper
function bullets(s, items, x, y, w, h) {
  const runs = items.map((t, i) => ({
    text: t,
    options: { bullet: true, breakLine: i < items.length - 1 },
  }));
  s.addText(runs, {
    x, y, w, h,
    fontSize: 13, fontFace: FONT_B, color: "1A2F4A",
    paraSpaceAfter: 4,
  });
}

// ── BUILD ─────────────────────────────────────────────────────────────────────
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title  = "Chennai Groundwater AI";
pres.author = "Final Year Engineering Project";

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 1 — TITLE                                                ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = darkSlide(pres);

  // Top accent strip
  s.addShape("rect", { x: 0, y: 0, w: 10, h: 0.12, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });

  // Left bold sidebar
  s.addShape("rect", { x: 0, y: 0.12, w: 0.18, h: 5.505, fill: { color: C.teal }, line: { color: C.teal, width: 0 } });

  // Decorative water-drop circles
  for (const [cx, cy, r, op] of [[8.5,1.2,1.2,8],[9.4,2.6,0.8,6],[7.8,3.8,0.5,4]]) {
    s.addShape("ellipse", {
      x: cx - r/2, y: cy - r/2, w: r, h: r,
      fill: { color: C.teal, transparency: 100 - op },
      line: { color: C.accent, width: 1.5, transparency: 100 - op * 2 },
    });
  }

  s.addText("AI-Powered", {
    x: 0.45, y: 0.7, w: 7.5, h: 0.55, margin: 0,
    fontSize: 16, fontFace: FONT_B, color: C.accent, bold: false, charSpacing: 4,
  });
  s.addText("Groundwater Prediction\n& Risk Assessment", {
    x: 0.45, y: 1.18, w: 7.8, h: 1.8, margin: 0,
    fontSize: 40, fontFace: FONT_H, bold: true, color: C.white,
  });

  // Tag line
  s.addShape("rect", { x: 0.45, y: 3.1, w: 4.8, h: 0.04, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  s.addText("System for Chennai", {
    x: 0.45, y: 3.22, w: 7, h: 0.45, margin: 0,
    fontSize: 20, fontFace: FONT_H, italic: true, color: "A8CDDF",
  });

  // Pill badges
  const badges = ["Machine Learning", "Deep Learning", "GIS Mapping", "Time-Series"];
  badges.forEach((b, i) => {
    const bx = 0.45 + i * 2.35;
    s.addShape("rect", { x: bx, y: 4.0, w: 2.15, h: 0.38,
      fill: { color: C.teal }, line: { color: C.accent, width: 1 } });
    s.addText(b, { x: bx, y: 4.0, w: 2.15, h: 0.38, margin: 0,
      fontSize: 9.5, fontFace: FONT_B, color: C.white, align: "center", valign: "middle" });
  });

  s.addText("Final Year Engineering Project  •  2024–25", {
    x: 0.45, y: 4.65, w: 7, h: 0.35, margin: 0,
    fontSize: 10, fontFace: FONT_B, color: C.muted,
  });

  // Bottom accent
  s.addShape("rect", { x: 0, y: 5.505, w: 10, h: 0.12, fill: { color: C.blue }, line: { color: C.blue, width: 0 } });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 2 — PROBLEM STATEMENT                                    ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Problem Statement");
  footer(s, true);

  // Two column layout
  // Left: text
  card(s, 0.35, 0.95, 5.2, 3.9, C.card, C.high);
  s.addText("The Crisis", {
    x: 0.55, y: 1.05, w: 4.8, h: 0.45, margin: 0,
    fontSize: 17, fontFace: FONT_H, bold: true, color: C.navy,
  });
  bullets(s, [
    "Chennai faces severe groundwater depletion due to rapid urbanisation and erratic monsoons",
    "Northeast monsoon (Oct–Dec) is the sole recharge window for most aquifers",
    "Coastal zones suffer saline intrusion reducing usable water quality",
    "No real-time AI system exists to forecast depletion and alert authorities",
    "Population density growth outpaces sustainable extraction rates",
  ], 0.55, 1.6, 4.85, 2.95);

  // Right: 3 stat cards stacked
  const stats = [["#1 Most", "Water-Stressed Metro in India"],
                 ["40%", "Annual Groundwater Deficit"],
                 ["2× faster", "Depletion vs Recharge Rate"]];
  stats.forEach(([v, l], i) => {
    statBox(s, 5.9, 0.95 + i * 1.32, v, l, [C.high, C.med, C.accent][i]);
  });

  // Quote
  s.addShape("rect", { x: 5.9, y: 4.92, w: 3.75, h: 0.7,
    fill: { color: C.navy }, line: { color: C.navy, width: 0 } });
  s.addText('"Without intervention, Chennai could face\na Day Zero water crisis by 2030."', {
    x: 5.9, y: 4.92, w: 3.75, h: 0.7, margin: 0,
    fontSize: 8.5, fontFace: FONT_B, italic: true, color: "A8CDDF",
    align: "center", valign: "middle",
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 3 — OBJECTIVES                                           ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Project Objectives");
  footer(s, true);

  const objs = [
    ["01", "Predict", "Forecast groundwater levels (continuous regression) for 10 Chennai zones using multi-source environmental data"],
    ["02", "Classify", "Categorise depletion risk as Low / Medium / High using ML classification models"],
    ["03", "Visualise", "Generate interactive GIS maps, IDW spatial interpolation, and heatmaps of Chennai"],
    ["04", "Explain", "Provide SHAP-based explainability so stakeholders understand model decisions"],
    ["05", "Deploy", "Deliver an interactive Streamlit web app for real-time predictions and scenario simulation"],
    ["06", "Project", "Simulate 30-year groundwater projections under IPCC RCP climate change scenarios"],
  ];

  objs.forEach(([num, title, desc], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = col === 0 ? 0.35 : 5.2;
    const y = 0.9 + row * 1.42;
    card(s, x, y, 4.6, 1.28, C.card, C.accent);
    s.addText(num, {
      x: x + 0.18, y: y + 0.1, w: 0.55, h: 0.55, margin: 0,
      fontSize: 22, fontFace: FONT_H, bold: true, color: C.blue,
    });
    s.addText(title, {
      x: x + 0.78, y: y + 0.1, w: 3.6, h: 0.38, margin: 0,
      fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy,
    });
    s.addText(desc, {
      x: x + 0.78, y: y + 0.48, w: 3.65, h: 0.7, margin: 0,
      fontSize: 10.5, fontFace: FONT_B, color: C.muted,
    });
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 4 — SYSTEM ARCHITECTURE                                  ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "System Architecture");
  footer(s, true);

  const layers = [
    ["DATA LAYER", "Rainfall · Temperature · NDVI · Population · Soil · Lat/Lon · Urban Fraction", C.blue],
    ["FEATURE ENGINEERING", "Lag Features (t-1,3,6,12) · Rolling Stats · SPI Drought Index · Urbanisation Index", C.teal],
    ["MODEL LAYER", "Ridge · Random Forest · XGBoost · LSTM · GRU · ARIMA · Hybrid Physics+ML", C.navy],
    ["OUTPUT LAYER", "GW Level Forecast · Risk Class · SHAP Explanation · GIS Map · Anomaly Alert", C.accent],
  ];

  layers.forEach(([title, desc, col], i) => {
    const y = 0.88 + i * 1.09;
    s.addShape("rect", { x: 0.35, y, w: 9.3, h: 0.92,
      fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(title, {
      x: 0.5, y, w: 2.3, h: 0.92, margin: 0,
      fontSize: 11, fontFace: FONT_H, bold: true, color: C.white, valign: "middle",
    });
    // divider
    s.addShape("rect", { x: 2.85, y: y + 0.18, w: 0.03, h: 0.56,
      fill: { color: C.white, transparency: 60 }, line: { color: C.white, width: 0 } });
    s.addText(desc, {
      x: 3.0, y, w: 6.45, h: 0.92, margin: 0,
      fontSize: 11.5, fontFace: FONT_B, color: C.white, valign: "middle",
    });
    // arrow down (not after last)
    if (i < layers.length - 1) {
      s.addShape("rect", { x: 4.87, y: y + 0.92, w: 0.26, h: 0.14,
        fill: { color: C.muted }, line: { color: C.muted, width: 0 } });
    }
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 5 — DATASET                                              ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Dataset & Data Sources");
  footer(s, true);

  // Left column: dataset specs
  card(s, 0.35, 0.9, 4.3, 4.1, C.card, C.teal);
  s.addText("Synthetic Multi-Source Dataset", {
    x: 0.55, y: 1.0, w: 3.95, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy,
  });

  const specs = [
    ["Zones", "10 Chennai administrative zones"],
    ["Period", "2000 – 2023 (monthly)"],
    ["Records", "2,880 rows × 16 features"],
    ["Target (Reg.)", "Groundwater level (m)"],
    ["Target (Cls.)", "Risk: Low / Medium / High"],
  ];
  specs.forEach(([k, v], i) => {
    const ry = 1.55 + i * 0.56;
    s.addText(k + ":", {
      x: 0.55, y: ry, w: 1.35, h: 0.38, margin: 0,
      fontSize: 11.5, fontFace: FONT_B, bold: true, color: C.teal,
    });
    s.addText(v, {
      x: 1.92, y: ry, w: 2.5, h: 0.38, margin: 0,
      fontSize: 11.5, fontFace: FONT_B, color: "1A2F4A",
    });
  });

  // Right column: feature grid
  card(s, 4.9, 0.9, 4.75, 4.1, C.card, C.accent);
  s.addText("Feature Categories", {
    x: 5.1, y: 1.0, w: 4.35, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy,
  });

  const feats = [
    ["Climate", "Rainfall, Temp, Evapotranspiration"],
    ["Vegetation", "NDVI (satellite proxy)"],
    ["Soil", "Type + Permeability coefficient"],
    ["Spatial", "Latitude, Longitude, Coastal flag"],
    ["Socio-Economic", "Population density, Urban fraction"],
    ["Engineered", "Lags, Rolling stats, SPI, Urb. Index"],
  ];
  feats.forEach(([cat, detail], i) => {
    const fy = 1.55 + i * 0.54;
    s.addShape("rect", { x: 5.1, y: fy + 0.06, w: 0.22, h: 0.22,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    s.addText(cat + ": ", {
      x: 5.42, y: fy, w: 1.25, h: 0.34, margin: 0,
      fontSize: 11, fontFace: FONT_B, bold: true, color: C.blue,
    });
    s.addText(detail, {
      x: 6.68, y: fy, w: 2.75, h: 0.34, margin: 0,
      fontSize: 11, fontFace: FONT_B, color: "1A2F4A",
    });
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 6 — FEATURE ENGINEERING                                  ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Feature Engineering");
  footer(s, true);

  const feCards = [
    ["Lag Features", "t-1, t-3, t-6, t-12 months for GW level and rainfall to capture memory effects", C.blue],
    ["Rolling Stats", "3, 6, 12-month moving averages and std-dev for rainfall and temperature", C.teal],
    ["SPI Drought Index", "Standardised Precipitation Index — z-score of rainfall per calendar month across all years", C.accent],
    ["Urbanisation Index", "Composite of urban fraction (50%), population density (30%) and inverted NDVI (20%)", C.navy],
    ["Seasonal Encoding", "Cyclical sin/cos of month + explicit season flag (NE monsoon / SW monsoon / dry)", C.high],
    ["Darcy Recharge", "Physics-derived estimate: (1–urban) × soil_permeability × rainfall × 0.002", C.low],
  ];

  feCards.forEach(([title, desc, col], i) => {
    const c = i % 3, r = Math.floor(i / 3);
    const x = 0.3 + c * 3.22, y = 0.88 + r * 1.65;
    card(s, x, y, 3.05, 1.48, C.card);
    s.addShape("rect", { x, y, w: 3.05, h: 0.38,
      fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(title, {
      x: x + 0.12, y, w: 2.85, h: 0.38, margin: 0,
      fontSize: 11.5, fontFace: FONT_H, bold: true, color: C.white, valign: "middle",
    });
    s.addText(desc, {
      x: x + 0.12, y: y + 0.44, w: 2.85, h: 0.94, margin: 0,
      fontSize: 10.5, fontFace: FONT_B, color: "1A2F4A",
    });
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 7 — ML MODELS                                            ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Machine Learning Models");
  footer(s, true);

  // Model comparison table
  const rows = [
    [{ text: "Model", options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "Type", options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "RMSE", options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "R²", options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "Key Strength", options: { bold: true, color: C.white, fill: { color: C.navy } } }],
    ["Ridge Regression", "Linear", "0.228", "0.9985", "Fast baseline, interpretable"],
    ["Random Forest", "Ensemble", "0.773", "0.9829", "Feature importance, robust"],
    [{ text: "XGBoost ★", options: { bold: true, color: C.navy } },
     "Gradient Boosting", "0.275", "0.9978", "Best overall accuracy"],
    ["Gradient Boosting", "Ensemble", "0.278", "0.9978", "Smooth predictions"],
    ["Logistic Regression", "Classifier", "—", "F1: 0.97 CV", "Risk classification"],
    ["RF Classifier", "Classifier", "—", "F1: 0.97 CV", "Risk + feature imp."],
    ["XGB Classifier", "Classifier", "—", "F1: 0.97 CV", "Risk, best CV score"],
  ];

  s.addTable(rows, {
    x: 0.35, y: 0.88, w: 9.3, h: 4.15,
    colW: [2.1, 1.7, 0.85, 1.0, 3.65],
    border: { pt: 0.5, color: "D0DDE8" },
    rowH: 0.46,
    fontSize: 11, fontFace: FONT_B, color: "1A2F4A",
    align: "center",
    autoPage: false,
    fill: { color: C.card },
    // alternate row shading via cell options below handled by fill on data rows
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 8 — DEEP LEARNING (LSTM/GRU)                             ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Deep Learning — LSTM & GRU (Pure NumPy)");
  footer(s, true);

  // Architecture diagram (left)
  card(s, 0.35, 0.9, 4.55, 4.1, C.card, C.blue);
  s.addText("Architecture", {
    x: 0.55, y: 1.0, w: 4.1, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy,
  });

  const archSteps = [
    ["Input", "12-month sliding window × 6 features"],
    ["LSTM Cell", "32 hidden units — sigmoid/tanh gates"],
    ["BPTT", "Backprop through time, gradient clipping ±5"],
    ["Output", "Dense → 3-step multi-step forecast"],
    ["Optimiser", "Adam (lr=0.001, β₁=0.9, β₂=0.999)"],
    ["Early Stop", "Patience=10, restore best weights"],
  ];
  archSteps.forEach(([k, v], i) => {
    const ay = 1.58 + i * 0.54;
    s.addShape("rect", { x: 0.55, y: ay + 0.06, w: 0.25, h: 0.25,
      fill: { color: C.blue }, line: { color: C.blue, width: 0 } });
    s.addText(k + ":", { x: 0.9, y: ay, w: 1.1, h: 0.36, margin: 0,
      fontSize: 10.5, fontFace: FONT_B, bold: true, color: C.blue });
    s.addText(v, { x: 2.05, y: ay, w: 2.65, h: 0.36, margin: 0,
      fontSize: 10.5, fontFace: FONT_B, color: "1A2F4A" });
  });

  // Results (right)
  card(s, 5.15, 0.9, 4.5, 4.1, C.card, C.teal);
  s.addText("Sample Results (Best Zones)", {
    x: 5.35, y: 1.0, w: 4.1, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy,
  });

  const results = [
    ["Zone", "LSTM R²", "GRU R²"],
    ["Adyar", "0.497", "0.786"],
    ["Anna Nagar", "0.298", "0.271"],
    ["Chromepet", "0.217", "0.724"],
    ["Porur", "−0.073", "0.753"],
  ];
  results.forEach(([zone, lstm, gru], i) => {
    const ry = 1.58 + i * 0.62;
    const isHeader = i === 0;
    s.addShape("rect", { x: 5.35, y: ry, w: 4.1, h: 0.52,
      fill: { color: isHeader ? C.teal : (i % 2 === 0 ? C.light : C.card) },
      line: { color: "D0DDE8", width: 0.5 } });
    [zone, lstm, gru].forEach((val, ci) => {
      s.addText(val, { x: 5.35 + ci * 1.37, y: ry, w: 1.37, h: 0.52, margin: 0,
        fontSize: 11, fontFace: FONT_B,
        bold: isHeader, color: isHeader ? C.white : "1A2F4A",
        align: "center", valign: "middle" });
    });
  });

  s.addText("Note: Pure NumPy LSTM — no TensorFlow/PyTorch required.\nAll 10 zones trained with early stopping.", {
    x: 5.35, y: 4.42, w: 4.1, h: 0.48, margin: 0,
    fontSize: 9, fontFace: FONT_B, italic: true, color: C.muted,
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 9 — ARIMA & TIME-SERIES                                  ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Time-Series Forecasting — ARIMA Baseline");
  footer(s, true);

  // Left: ARIMA process
  card(s, 0.35, 0.9, 4.4, 4.1, C.card, C.teal);
  s.addText("ARIMA Pipeline", {
    x: 0.55, y: 1.0, w: 4.0, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy,
  });
  const steps = [
    "1  ADF Stationarity Test per zone",
    "2  Auto-select (p, d, q) by AIC grid search",
    "3  Fit SARIMAX model on 80% train split",
    "4  Forecast 20% test horizon",
    "5  Generate 24-month future projections",
    "6  Save diagnostic residual plots",
  ];
  steps.forEach((st, i) => {
    s.addText(st, { x: 0.55, y: 1.58 + i * 0.56, w: 3.95, h: 0.42, margin: 0,
      fontSize: 11.5, fontFace: FONT_B, color: "1A2F4A" });
  });

  // Right: sample zone results
  card(s, 5.05, 0.9, 4.6, 4.1, C.card, C.accent);
  s.addText("Zone Results (ARIMA)", {
    x: 5.25, y: 1.0, w: 4.2, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy,
  });

  const arRes = [
    ["Zone", "Order", "RMSE"],
    ["Adyar", "(2,0,2)", "0.510"],
    ["Anna Nagar", "(2,1,2)", "0.074"],
    ["Chromepet", "(2,1,2)", "0.090"],
    ["Velachery", "(2,1,2)", "0.217"],
    ["Tambaram", "(2,0,0)", "0.823"],
  ];
  arRes.forEach(([zone, order, rmse], i) => {
    const ry = 1.58 + i * 0.55;
    const isHeader = i === 0;
    s.addShape("rect", { x: 5.25, y: ry, w: 4.2, h: 0.46,
      fill: { color: isHeader ? C.navy : (i % 2 === 0 ? C.light : C.card) },
      line: { color: "D0DDE8", width: 0.5 } });
    [zone, order, rmse].forEach((val, ci) => {
      s.addText(val, { x: 5.25 + ci * 1.4, y: ry, w: 1.4, h: 0.46, margin: 0,
        fontSize: 11, fontFace: FONT_B,
        bold: isHeader, color: isHeader ? C.white : "1A2F4A",
        align: "center", valign: "middle" });
    });
  });

  s.addText("ARIMA serves as interpretable baseline.\nML models significantly outperform on all zones.", {
    x: 5.25, y: 4.5, w: 4.2, h: 0.38, margin: 0,
    fontSize: 9.5, fontFace: FONT_B, italic: true, color: C.muted,
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 10 — GIS & SPATIAL ANALYSIS                              ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "GIS & Spatial Analysis");
  footer(s, true);

  const gisCards = [
    ["Interactive Folium Map", "Zone markers colour-coded by risk level\n(Low/Medium/High) with popup stats and\ndepletion heatmap overlay", C.blue],
    ["IDW Interpolation", "Inverse Distance Weighting contour map\ninterpolating groundwater levels across\nChennai's spatial grid (200×200)", C.teal],
    ["Risk Heatmap", "Bar chart of risk distribution across zones\n+ spatial scatter plot of all zones coloured\nby depletion risk category", C.high],
    ["Trend Map", "Annual mean groundwater level per zone\nplotted 2000–2023 showing long-term\ndepletion trajectory per zone", C.accent],
  ];

  gisCards.forEach(([title, desc, col], i) => {
    const x = 0.35 + (i % 2) * 4.85;
    const y = 0.88 + Math.floor(i / 2) * 2.2;
    card(s, x, y, 4.55, 2.0, C.card);
    s.addShape("rect", { x, y, w: 4.55, h: 0.5,
      fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(title, { x: x + 0.14, y, w: 4.3, h: 0.5, margin: 0,
      fontSize: 13, fontFace: FONT_H, bold: true, color: C.white, valign: "middle" });
    s.addText(desc, { x: x + 0.14, y: y + 0.58, w: 4.3, h: 1.3, margin: 0,
      fontSize: 11.5, fontFace: FONT_B, color: "1A2F4A" });
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 11 — RISK CLASSIFICATION + SHAP                          ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Risk Classification & Explainable AI (SHAP)");
  footer(s, true);

  // Risk classes
  const riskCards = [
    ["LOW RISK", "GW depth < 12 m\nAdequate recharge\nSustainable extraction", C.low, "27AE60"],
    ["MEDIUM RISK", "GW depth 12–20 m\nRecharge deficit emerging\nMonitoring required", C.med, "F39C12"],
    ["HIGH RISK", "GW depth > 20 m\nSevere depletion\nImmediate intervention", C.high, "E74C3C"],
  ];
  riskCards.forEach(([title, desc, col], i) => {
    const x = 0.35 + i * 3.15;
    s.addShape("rect", { x, y: 0.88, w: 3.0, h: 0.55,
      fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(title, { x, y: 0.88, w: 3.0, h: 0.55, margin: 0,
      fontSize: 13, fontFace: FONT_H, bold: true, color: C.white,
      align: "center", valign: "middle" });
    card(s, x, 1.43, 3.0, 1.5, C.card);
    s.addText(desc, { x: x + 0.12, y: 1.55, w: 2.78, h: 1.28, margin: 0,
      fontSize: 12, fontFace: FONT_B, color: "1A2F4A" });
  });

  // Classifiers + metrics
  card(s, 0.35, 3.1, 5.9, 1.8, C.card, C.navy);
  s.addText("Classifier Performance", {
    x: 0.55, y: 3.2, w: 5.5, h: 0.4, margin: 0,
    fontSize: 13, fontFace: FONT_H, bold: true, color: C.navy });
  bullets(s, [
    "Logistic Regression  — CV F1 macro: 0.968  |  Solver: SAGA",
    "Random Forest Classifier  — CV F1 macro: 0.970",
    "XGBoost Classifier  — CV F1 macro: 0.974  (best)",
  ], 0.55, 3.68, 5.6, 1.12);

  // SHAP
  card(s, 6.5, 3.1, 3.15, 1.8, C.card, C.accent);
  s.addText("SHAP Explainability", {
    x: 6.7, y: 3.2, w: 2.75, h: 0.4, margin: 0,
    fontSize: 13, fontFace: FONT_H, bold: true, color: C.navy });
  s.addText("• TreeExplainer for RF & XGBoost\n• LinearExplainer for Logistic Regression\n• Top-20 feature importance summary plots\n• Applied to High Risk class output", {
    x: 6.7, y: 3.68, w: 2.75, h: 1.12, margin: 0,
    fontSize: 10.5, fontFace: FONT_B, color: "1A2F4A" });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 12 — ADVANCED MODULES                                    ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Advanced Enhancement Modules");
  footer(s, true);

  const mods = [
    {
      title: "Hybrid Physics + ML",
      color: C.blue,
      points: [
        "Physics layer: Darcy's Law recharge estimate",
        "R = (1−urban) × soil_K × rainfall × 0.002",
        "ML layer (Random Forest) corrects physics residuals",
        "Combined model R² = 0.70 on held-out test set",
      ],
    },
    {
      title: "Climate Change Scenarios",
      color: C.teal,
      points: [
        "RCP 2.6 — Optimistic: +5% rainfall, +1°C",
        "RCP 4.5 — Moderate: −10% rainfall, +2°C",
        "RCP 8.5 — Pessimistic: −20% rainfall, +3.5°C",
        "30-year projection per zone, interactive chart",
      ],
    },
    {
      title: "Anomaly Detection",
      color: C.high,
      points: [
        "Layer 1: Isolation Forest (contamination=5%)",
        "Layer 2: Rolling z-score threshold (default=3σ)",
        "Combined flag: either layer triggers alert",
        "9–21 anomalies detected per zone (2000–2023)",
      ],
    },
  ];

  mods.forEach(({ title, color, points }, i) => {
    const x = 0.35 + i * 3.22;
    card(s, x, 0.88, 3.05, 4.1, C.card);
    s.addShape("rect", { x, y: 0.88, w: 3.05, h: 0.55,
      fill: { color }, line: { color, width: 0 } });
    s.addText(title, { x: x + 0.1, y: 0.88, w: 2.88, h: 0.55, margin: 0,
      fontSize: 12.5, fontFace: FONT_H, bold: true, color: C.white, valign: "middle" });
    const runs = points.map((p, pi) => ({
      text: p, options: { bullet: true, breakLine: pi < points.length - 1 },
    }));
    s.addText(runs, { x: x + 0.14, y: 1.55, w: 2.78, h: 3.28,
      fontSize: 11, fontFace: FONT_B, color: "1A2F4A", paraSpaceAfter: 5 });
  });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 13 — WEB APPLICATION                                     ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Interactive Web Application — Streamlit");
  footer(s, true);

  // 7 page cards in a grid
  const pages = [
    ["Dashboard", "KPIs, trend chart, risk pie chart", C.navy],
    ["Prediction", "Input params → GW gauge + risk bars", C.blue],
    ["Analytics", "Time series, correlation, seasonal", C.teal],
    ["GIS Maps", "Folium map + IDW + heatmap", C.accent],
    ["Risk Assessment", "Zone cards + annual bar chart", C.high],
    ["Climate Scenarios", "RCP 2.6/4.5/8.5 projections", C.med],
    ["Anomaly Detection", "Isolation Forest + z-score chart", C.low],
  ];

  pages.forEach(([title, desc, col], i) => {
    const col_i = i % 4, row_i = Math.floor(i / 4);
    const x = 0.35 + col_i * 2.37;
    const y = 0.9 + row_i * 1.7;
    card(s, x, y, 2.2, 1.52, C.card);
    s.addShape("rect", { x, y, w: 2.2, h: 0.42,
      fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(title, { x: x + 0.1, y, w: 2.05, h: 0.42, margin: 0,
      fontSize: 11, fontFace: FONT_H, bold: true, color: C.white, valign: "middle" });
    s.addText(desc, { x: x + 0.1, y: y + 0.5, w: 2.05, h: 0.9, margin: 0,
      fontSize: 10, fontFace: FONT_B, color: "1A2F4A" });
  });

  // Tech stack strip
  s.addShape("rect", { x: 0.35, y: 4.72, w: 9.3, h: 0.42,
    fill: { color: C.navy }, line: { color: C.navy, width: 0 } });
  s.addText("Tech Stack:  Streamlit  ·  scikit-learn  ·  XGBoost  ·  statsmodels  ·  Folium  ·  SHAP  ·  Pure NumPy LSTM  ·  Matplotlib", {
    x: 0.35, y: 4.72, w: 9.3, h: 0.42, margin: 0,
    fontSize: 10, fontFace: FONT_B, color: C.accent, align: "center", valign: "middle" });
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 14 — RESULTS SUMMARY                                     ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = lightSlide(pres);
  sectionBar(s, "Results Summary");
  footer(s, true);

  // Best scores
  const topStats = [
    ["0.9985", "Best R²\n(Ridge Regression)", C.low],
    ["0.974", "Best Classifier\nF1-macro (XGBoost)", C.blue],
    ["63", "Models Trained\n& Saved", C.teal],
    ["2,880", "Dataset Records\n10 Zones × 24 Yrs", C.navy],
  ];
  topStats.forEach(([v, l, col], i) => statBox(s, 0.35 + i * 2.38, 0.9, v, l, col));

  // Key findings
  card(s, 0.35, 2.22, 5.6, 2.68, C.card, C.blue);
  s.addText("Key Findings", {
    x: 0.55, y: 2.32, w: 5.2, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy });
  bullets(s, [
    "XGBoost and Ridge Regression achieve R² > 0.99 on regression task",
    "GRU outperforms LSTM on 6 of 10 zones for time-series forecasting",
    "ARIMA provides interpretable baseline — ML models dominate accuracy",
    "RCP 8.5 scenario projects further 5–12 m depletion by 2053",
    "9–21 anomalies detected per zone using dual-layer anomaly detection",
  ], 0.55, 2.82, 5.25, 1.9);

  // Future scope
  card(s, 6.2, 2.22, 3.45, 2.68, C.card, C.accent);
  s.addText("Future Scope", {
    x: 6.4, y: 2.32, w: 3.05, h: 0.42, margin: 0,
    fontSize: 14, fontFace: FONT_H, bold: true, color: C.navy });
  bullets(s, [
    "Real satellite data (GRACE)",
    "Transformer-based forecast",
    "IoT sensor integration",
    "Government API deployment",
    "Multi-city generalisation",
  ], 6.4, 2.82, 3.05, 1.9);
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  SLIDE 15 — CONCLUSION (DARK)                                   ║
// ╚══════════════════════════════════════════════════════════════════╝
{
  const s = darkSlide(pres);

  s.addShape("rect", { x: 0, y: 0, w: 10, h: 0.1, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  s.addShape("rect", { x: 0, y: 5.525, w: 10, h: 0.1, fill: { color: C.teal }, line: { color: C.teal, width: 0 } });

  s.addText("Conclusion", {
    x: 0.6, y: 0.4, w: 8.8, h: 0.7, margin: 0,
    fontSize: 34, fontFace: FONT_H, bold: true, color: C.white, align: "center",
  });

  s.addShape("rect", { x: 3.5, y: 1.12, w: 3.0, h: 0.04, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });

  const conclusions = [
    "Built a fully modular, production-ready AI system for Chennai groundwater prediction",
    "Integrated 7 model types: Ridge, RF, XGBoost, GBM, LSTM, GRU, ARIMA — all running without TensorFlow",
    "Hybrid Physics+ML model incorporates Darcy's Law for domain-aware predictions",
    "GIS pipeline produces interactive Folium maps and IDW spatial interpolation of depletion zones",
    "SHAP explainability makes model decisions transparent to policymakers",
    "Streamlit web app provides real-time prediction, risk classification, and climate scenario simulation",
  ];

  const runs = conclusions.map((c, i) => ({
    text: c, options: { bullet: true, breakLine: i < conclusions.length - 1 },
  }));
  s.addText(runs, {
    x: 1.0, y: 1.3, w: 8.0, h: 3.4,
    fontSize: 13.5, fontFace: FONT_B, color: "C8DCF0", paraSpaceAfter: 6,
  });

  s.addText("Thank You", {
    x: 0.6, y: 4.82, w: 8.8, h: 0.52, margin: 0,
    fontSize: 22, fontFace: FONT_H, bold: true, italic: true,
    color: C.accent, align: "center",
  });
}

// ── Write file ────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "/Users/akshpuri/Desktop/final/Chennai_Groundwater_AI.pptx" })
  .then(() => console.log("✅  Chennai_Groundwater_AI.pptx saved"))
  .catch(err => { console.error("Error:", err); process.exit(1); });
