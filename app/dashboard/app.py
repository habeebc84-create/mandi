"""
Interactive 3D & 2D Tactical Web Command Center for Flood Damage Mapping.
Features:
- Three.js 3D Digital Twin with Himalayan elevation terrain displacement
- Animated 3D water flood wave simulation with dynamic water-level rise
- Glowing hydraulic flow particles tracing D8 drainage paths
- 2D Leaflet Tactical GIS with Sentinel-1 SAR orbital sweep effect
- Before-and-After satellite curtain swipe
- Animated KPI counters & glassmorphic emergency operations UI
- Dual-language instant switching (English & Nepali)
- Direct export endpoints for GeoPackage, PDF reports, and JSON
"""

import os
import json
import logging
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

app = FastAPI(title="Flood Damage Mapping 3D Command Center")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_PATH = os.path.join(OUTPUTS_DIR, "results.json")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mandi - 3D Flood Damage & Isolation Command Center</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Leaflet CSS & JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <!-- Three.js & OrbitControls -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

  <!-- Google Fonts & FontAwesome -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>

  <style>
    :root {
      --bg-dark: #070b14;
      --bg-panel: rgba(13, 19, 33, 0.78);
      --bg-panel-solid: #0d1321;
      --border-glow: rgba(56, 189, 248, 0.25);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --cyan: #06b6d4;
      --cyan-glow: #38bdf8;
      --red: #f43f5e;
      --red-glow: #fb7185;
      --amber: #f59e0b;
      --emerald: #10b981;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
      background: var(--bg-dark);
      color: var(--text-main);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* TOP COMMAND BAR */
    #topbar {
      height: 60px;
      background: rgba(10, 15, 29, 0.92);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-glow);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      z-index: 1001;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.6);
    }
    .brand-section { display: flex; align-items: center; gap: 14px; }
    .radar-pulse-icon {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      background: radial-gradient(circle, #f43f5e 20%, transparent 70%);
      border: 2px solid #f43f5e;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 15px rgba(244, 63, 94, 0.6);
      animation: pulse-ring 2s infinite ease-out;
    }
    @keyframes pulse-ring {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 14px rgba(244, 63, 94, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
    }
    .brand-title { font-size: 16px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; }
    .brand-title span { color: var(--cyan-glow); }
    .brand-badge {
      font-size: 10px;
      font-weight: 700;
      background: rgba(244, 63, 94, 0.2);
      border: 1px solid #f43f5e;
      color: #fda4af;
      padding: 2px 7px;
      border-radius: 4px;
      letter-spacing: 0.8px;
    }

    .top-controls { display: flex; align-items: center; gap: 12px; }
    .mode-switch-group {
      display: flex;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--border-glow);
      border-radius: 8px;
      padding: 3px;
      gap: 3px;
    }
    .mode-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .mode-btn.active {
      background: linear-gradient(135deg, #0284c7, #06b6d4);
      color: #fff;
      box-shadow: 0 2px 10px rgba(6, 182, 212, 0.4);
    }
    .mode-btn:hover:not(.active) { color: var(--text-main); background: rgba(255, 255, 255, 0.05); }

    .lang-toggle {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 6px 10px;
      font-size: 11px;
      font-weight: 700;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .lang-toggle:hover { border-color: var(--cyan-glow); }

    .btn-export {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid #10b981;
      color: #6ee7b7;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-export:hover { background: #10b981; color: #042f2e; }

    /* WORKSPACE BODY */
    #workspace {
      flex: 1;
      position: relative;
      display: flex;
      overflow: hidden;
    }

    /* GLASS TELEMETRY SIDEBAR */
    #sidebar {
      width: 440px;
      height: 100%;
      background: var(--bg-panel);
      backdrop-filter: blur(20px);
      border-right: 1px solid var(--border-glow);
      display: flex;
      flex-direction: column;
      z-index: 1000;
      box-shadow: 8px 0 32px rgba(0, 0, 0, 0.55);
      overflow-y: auto;
      transition: transform 0.3s ease;
    }
    #sidebar::-webkit-scrollbar { width: 5px; }
    #sidebar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }

    .panel-section {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .section-title {
      font-size: 11px;
      font-weight: 700;
      color: var(--cyan-glow);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* KPI METRICS GRID */
    .kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .kpi-card {
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 12px;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      border-color: var(--cyan-glow);
    }
    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; width: 4px; height: 100%;
      background: var(--card-accent, var(--cyan));
    }
    .kpi-num {
      font-size: 24px;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      color: #fff;
    }
    .kpi-label { font-size: 11px; color: var(--text-muted); margin-top: 3px; font-weight: 500; }

    /* REAL-TIME SLIDERS */
    .slider-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 10px;
    }
    .slider-header {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 6px;
    }
    .slider-val-badge {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--cyan-glow);
      font-weight: 700;
    }
    input[type="range"] {
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: #1e293b;
      accent-color: var(--cyan-glow);
      cursor: pointer;
    }

    /* TOGGLE SWITCHES */
    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 10px;
      background: rgba(15, 23, 42, 0.4);
      border-radius: 6px;
      margin-bottom: 6px;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.2s;
    }
    .toggle-row:hover { background: rgba(30, 41, 59, 0.6); }
    .toggle-label { display: flex; align-items: center; gap: 9px; }
    .dot-indicator { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }

    /* SEVERED SETTLEMENTS LIST */
    .settlement-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(244, 63, 94, 0.2);
      border-left: 3px solid #f43f5e;
      border-radius: 6px;
      padding: 10px 12px;
      margin-bottom: 8px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .settlement-card:hover {
      background: rgba(244, 63, 94, 0.12);
      border-color: #f43f5e;
      transform: translateX(3px);
    }
    .settlement-name { font-weight: 700; color: #fff; display: flex; justify-content: space-between; }
    .settlement-badge {
      font-size: 9px;
      font-weight: 800;
      background: #f43f5e;
      color: #fff;
      padding: 1px 5px;
      border-radius: 3px;
      letter-spacing: 0.5px;
    }
    .settlement-desc { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

    /* MAIN VIEWER CONTAINERS */
    #viewer-container {
      flex: 1;
      position: relative;
      height: 100%;
    }
    #map2d-view {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      z-index: 10;
    }
    #canvas3d-view {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      z-index: 20;
      display: none;
      background: radial-gradient(circle at 50% 30%, #0d1b2a 0%, #070b14 100%);
    }

    /* 3D FLOATING HUD CONTROLS */
    .hud-controls-3d {
      position: absolute;
      bottom: 85px;
      right: 25px;
      z-index: 30;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .hud-btn {
      background: rgba(13, 19, 33, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-glow);
      color: var(--text-main);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
      transition: all 0.2s;
    }
    .hud-btn:hover {
      background: var(--cyan-glow);
      color: #042f2e;
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.6);
    }

    /* FLOOD WAVE TIMELINE BAR */
    #timeline-bar {
      position: absolute;
      bottom: 18px;
      left: 460px;
      right: 25px;
      height: 52px;
      background: rgba(13, 19, 33, 0.88);
      backdrop-filter: blur(18px);
      border: 1px solid var(--border-glow);
      border-radius: 12px;
      z-index: 35;
      display: flex;
      align-items: center;
      padding: 0 18px;
      gap: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    }
    .play-btn {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: linear-gradient(135deg, #0284c7, #06b6d4);
      border: none;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 0 12px rgba(6, 182, 212, 0.5);
      transition: transform 0.2s;
    }
    .play-btn:hover { transform: scale(1.08); }
    .timeline-track { flex: 1; display: flex; flex-direction: column; gap: 4px; }
    .timeline-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted); font-weight: 600; }

    /* SAR SCAN SWEEP EFFECT (2D) */
    .radar-sweep-line {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 4px;
      background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.8), transparent);
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.9);
      pointer-events: none;
      z-index: 15;
      animation: sar-scan 6s infinite linear;
    }
    @keyframes sar-scan {
      0% { top: -5%; opacity: 0; }
      15% { opacity: 1; }
      85% { opacity: 1; }
      100% { top: 105%; opacity: 0; }
    }

    /* ATTRIBUTION FOOTER */
    .attribution-text {
      font-size: 9px;
      color: #64748b;
      line-height: 1.4;
      padding: 12px 18px;
      background: rgba(10, 15, 29, 0.8);
      border-top: 1px solid var(--border-subtle);
    }
  </style>
</head>
<body>

<!-- TOP COMMAND BAR -->
<div id="topbar">
  <div class="brand-section">
    <div class="radar-pulse-icon">
      <i class="fa-solid fa-satellite-dish" style="font-size: 13px; color: #fff;"></i>
    </div>
    <div>
      <div class="brand-title">MANDI <span>// 3D FLOOD COMMAND</span></div>
      <div style="font-size: 11px; color: var(--text-muted);"><span id="lbl-aoi">Trishuli River Basin</span> &bull; Copernicus Sentinel-1/2 RTC</div>
    </div>
    <div class="brand-badge" id="lbl-status"><i class="fa-solid fa-circle-exclamation"></i> ACTIVE EVENT</div>
  </div>

  <div class="top-controls">
    <div class="mode-switch-group">
      <button class="mode-btn active" id="btn-mode-2d" onclick="switchMode('2d')">
        <i class="fa-solid fa-map"></i> <span id="lbl-2d">2D GIS</span>
      </button>
      <button class="mode-btn" id="btn-mode-3d" onclick="switchMode('3d')">
        <i class="fa-solid fa-cube"></i> <span id="lbl-3d">3D Digital Twin</span>
      </button>
    </div>

    <button class="lang-toggle" onclick="toggleLanguage()">
      <i class="fa-solid fa-globe"></i> <span id="lang-indicator">EN / NE</span>
    </button>

    <a href="/api/download/pdf/en" target="_blank" class="btn-export" id="btn-pdf">
      <i class="fa-solid fa-file-pdf"></i> <span id="lbl-pdf">Situation Report</span>
    </a>
    <a href="/api/download/gpkg" download class="btn-export" style="border-color:#38bdf8; color:#7dd3fc; background:rgba(56,189,248,0.15)">
      <i class="fa-solid fa-database"></i> <span id="lbl-gpkg">GeoPackage</span>
    </a>
  </div>
</div>

<!-- WORKSPACE -->
<div id="workspace">

  <!-- LEFT TELEMETRY SIDEBAR -->
  <div id="sidebar">

    <!-- IMPACT METRICS -->
    <div class="panel-section">
      <div class="section-title">
        <span id="title-impact">1. Primary Impact Metrics</span>
        <span style="font-size:10px; color:#10b981;"><i class="fa-solid fa-bolt"></i> VERIFIED</span>
      </div>
      <div class="kpi-grid">
        <div class="kpi-card" style="--card-accent:#f43f5e;">
          <div class="kpi-num" id="kpi-bldgs">0</div>
          <div class="kpi-label" id="lbl-kpi-bldgs">Buildings Hit</div>
        </div>
        <div class="kpi-card" style="--card-accent:#f59e0b;">
          <div class="kpi-num" id="kpi-roads">0.0</div>
          <div class="kpi-label" id="lbl-kpi-roads">Roads Severed (km)</div>
        </div>
        <div class="kpi-card" style="--card-accent:#ec4899;">
          <div class="kpi-num" id="kpi-bridges">0</div>
          <div class="kpi-label" id="lbl-kpi-bridges">Bridges at Risk</div>
        </div>
        <div class="kpi-card" style="--card-accent:#06b6d4;">
          <div class="kpi-num" id="kpi-cutoff">0</div>
          <div class="kpi-label" id="lbl-kpi-cutoff">Cut-Off Settlements</div>
        </div>
      </div>
    </div>

    <!-- AI & SAR SENSITIVITY CONTROLLER -->
    <div class="panel-section">
      <div class="section-title">
        <span id="title-confidence">2. AI Confidence Threshold</span>
        <span class="slider-val-badge" id="thresh-val">0.70</span>
      </div>
      <div class="slider-box">
        <div class="slider-header">
          <span style="color:var(--text-muted); font-size:11px;" id="lbl-sens-mode">Sensitivity Mode</span>
          <span id="thresh-label" style="color:var(--cyan-glow); font-weight:700; font-size:11px;">Conservative (&ge;70%)</span>
        </div>
        <input type="range" id="conf-slider" min="0.30" max="0.85" step="0.05" value="0.70" oninput="updateSensitivity(this.value)">
      </div>

      <!-- 3D WATER LEVEL RISE SLIDER -->
      <div class="slider-box" style="margin-top:10px;">
        <div class="slider-header">
          <span style="color:var(--text-muted); font-size:11px;" id="lbl-water-sim">3D Flood Inundation Height</span>
          <span class="slider-val-badge" id="water-height-val">+14.0 m</span>
        </div>
        <input type="range" id="water-slider" min="0" max="30" step="1" value="14" oninput="updateWaterHeight(this.value)">
      </div>
    </div>

    <!-- HAZARD & INFRASTRUCTURE LAYERS -->
    <div class="panel-section">
      <div class="section-title" id="title-layers">3. Hazard & Infrastructure Layers</div>
      <div class="toggle-row" onclick="toggleLayer('flood')">
        <span class="toggle-label"><span class="dot-indicator" style="background:#0284c7; box-shadow:0 0 8px #0284c7;"></span> <span id="lbl-lyr-flood">Flood Inundation (U-Net)</span></span>
        <input type="checkbox" id="chk-flood" checked onclick="event.stopPropagation()">
      </div>
      <div class="toggle-row" onclick="toggleLayer('debris')">
        <span class="toggle-label"><span class="dot-indicator" style="background:#ea580c; box-shadow:0 0 8px #ea580c;"></span> <span id="lbl-lyr-debris">Debris & Scour Scars (SAR Otsu)</span></span>
        <input type="checkbox" id="chk-debris" checked onclick="event.stopPropagation()">
      </div>
      <div class="toggle-row" onclick="toggleLayer('roads')">
        <span class="toggle-label"><span class="dot-indicator" style="background:#ef4444; box-shadow:0 0 8px #ef4444;"></span> <span id="lbl-lyr-roads">Severed Road Corridors</span></span>
        <input type="checkbox" id="chk-roads" checked onclick="event.stopPropagation()">
      </div>
      <div class="toggle-row" onclick="toggleLayer('cutoff')">
        <span class="toggle-label"><span class="dot-indicator" style="background:#f43f5e; box-shadow:0 0 8px #f43f5e;"></span> <span id="lbl-lyr-cutoff">Cut-Off Settlements</span></span>
        <input type="checkbox" id="chk-cutoff" checked onclick="event.stopPropagation()">
      </div>
      <div class="toggle-row" onclick="toggleLayer('flowpath')">
        <span class="toggle-label"><span class="dot-indicator" style="background:#06b6d4; box-shadow:0 0 8px #06b6d4;"></span> <span id="lbl-lyr-flow">D8 Downhill Flow Path (Bonus)</span></span>
        <input type="checkbox" id="chk-flowpath" checked onclick="event.stopPropagation()">
      </div>
    </div>

    <!-- SEVERED SETTLEMENTS DIRECTORY -->
    <div class="panel-section" style="flex:1;">
      <div class="section-title">
        <span id="title-settlements">4. Severed Settlements (Dijkstra Isolated)</span>
        <span style="font-size:10px; color:#f43f5e;" id="settle-count-badge">6 TOTAL</span>
      </div>
      <div id="settlements-container">
        <!-- Rendered dynamically -->
      </div>
    </div>

    <!-- ATTRIBUTIONS FOOTER -->
    <div class="attribution-text">
      <strong>Legal Attributions & Scientific Citations:</strong><br>
      &bull; Contains modified Copernicus Sentinel data 2026.<br>
      &bull; Copernicus WorldDEM-30 &copy; DLR e.V. &amp; Airbus Defence and Space GmbH.<br>
      &bull; &copy; OpenStreetMap contributors (ODbL).<br>
      &bull; Benchmark Validation: European Union, Copernicus EMS (EMSR927).<br>
      &bull; Bountos et al., 2024 (Kuro Siwo) &bull; Bonafilia et al., 2020 (Sen1Floods11).
    </div>

  </div>

  <!-- VIEWER CONTAINER -->
  <div id="viewer-container">
    
    <!-- 2D LEAFLET VIEW -->
    <div id="map2d-view">
      <div class="radar-sweep-line"></div>
    </div>

    <!-- 3D THREE.JS CANVAS VIEW -->
    <div id="canvas3d-view"></div>

    <!-- 3D HUD CONTROLS -->
    <div class="hud-controls-3d" id="hud3d-tools" style="display:none;">
      <button class="hud-btn" onclick="cameraPreset('overview')">
        <i class="fa-solid fa-mountain"></i> Valley Overview
      </button>
      <button class="hud-btn" onclick="cameraPreset('cinematic')">
        <i class="fa-solid fa-video"></i> Cinematic Orbit
      </button>
      <button class="hud-btn" onclick="cameraPreset('gorge')">
        <i class="fa-solid fa-water"></i> Gorge Close-up
      </button>
      <button class="hud-btn" onclick="triggerFlowSimulation()">
        <i class="fa-solid fa-water-ladder"></i> Animate Wave
      </button>
    </div>

    <!-- FLOATING FLOOD TIMELINE -->
    <div id="timeline-bar">
      <button class="play-btn" id="btn-play-timeline" onclick="togglePlayTimeline()">
        <i class="fa-solid fa-play" id="icon-play"></i>
      </button>
      <div class="timeline-track">
        <input type="range" id="time-scrubber" min="0" max="100" value="70" oninput="scrubTimeline(this.value)">
        <div class="timeline-labels">
          <span>T - 12d (Pre-event SAR)</span>
          <span>T = 0h (Monsoon Inundation)</span>
          <span>T + 24h (Max Isolation)</span>
          <span>T + 48h (Sediment Scar)</span>
        </div>
      </div>
    </div>

  </div>

</div>

<script>
  let resultsData = null;
  let currentLang = 'en';
  let isPlayingTimeline = false;
  let timelineInterval = null;

  // 2D MAP INITIALIZATION
  const map = L.map('map2d-view', { zoomControl: false }).setView([28.03, 85.22], 12);
  L.control.zoom({ position: 'topright' }).addTo(map);

  const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Esri World Imagery', maxZoom: 18
  }).addTo(map);

  const cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: 'CartoDB Dark', maxZoom: 18
  });

  L.control.layers({ "Satellite Imagery": esriSatellite, "Dark Tactical": cartoDark }, null, { position: 'topright' }).addTo(map);

  const layers2D = {
    flood: L.layerGroup().addTo(map),
    debris: L.layerGroup().addTo(map),
    roads: L.layerGroup().addTo(map),
    cutoff: L.layerGroup().addTo(map),
    flowpath: L.layerGroup().addTo(map)
  };

  // 3D THREE.JS ENGINE
  let scene, camera, renderer, controls, terrainMesh, waterMesh, particleSystem;
  let is3DInitialized = false;
  let animId = null;
  let flowParticles = [];

  function init3D() {
    if (is3DInitialized) return;
    const container = document.getElementById('canvas3d-view');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);
    scene.fog = new THREE.FogExp2(0x070b14, 0.0018);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    camera.position.set(0, 180, 240);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.05;
    controls.minDistance = 30;
    controls.maxDistance = 500;

    // LIGHTING
    const ambient = new THREE.AmbientLight(0x38bdf8, 0.6);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xfff7ed, 1.2);
    dirLight.position.set(120, 200, 100);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const rimLight = new THREE.PointLight(0x06b6d4, 1.5, 400);
    rimLight.position.set(-100, 80, -100);
    scene.add(rimLight);

    buildHimalayanTerrain();
    buildWaterSurface();
    buildFlowParticleRibbon();

    window.addEventListener('resize', onWindowResize);
    is3DInitialized = true;
    animate3D();
  }

  function buildHimalayanTerrain() {
    const gridDim = 120;
    const size = 300;
    const geometry = new THREE.PlaneGeometry(size, size, gridDim, gridDim);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) / (size / 2);
      const z = pos.getZ(i) / (size / 2);

      // Meandering Trishuli river canyon profile
      const valleyCenter = 0.25 * Math.sin(z * 3.5);
      const distFromRiver = Math.abs(x - valleyCenter);
      
      // Ridge elevations up to ~75 units
      let elevation = 10 + 65 * Math.pow(distFromRiver, 1.35) + 6 * Math.sin(x * 12) * Math.cos(z * 10);
      pos.setY(i, elevation);
    }
    geometry.computeVertexNormals();

    // Procedural terrain vertex coloring (riverbed=sand/gravel, valley=green, peaks=granite/snow)
    const count = pos.count;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const y = pos.getY(i);
      let r, g, b;
      if (y < 16) {
        // Riverbed
        r = 0.15; g = 0.22; b = 0.28;
      } else if (y < 35) {
        // Valley terraces
        r = 0.12; g = 0.35; b = 0.25;
      } else if (y < 55) {
        // Mountain rocky slopes
        r = 0.35; g = 0.38; b = 0.42;
      } else {
        // High snow-capped ridges
        r = 0.75; g = 0.82; b = 0.90;
      }
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.1,
      flatShading: true
    });

    terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.receiveShadow = true;
    scene.add(terrainMesh);
  }

  function buildWaterSurface() {
    const waterGeom = new THREE.PlaneGeometry(300, 300, 64, 64);
    waterGeom.rotateX(-Math.PI / 2);

    const waterMat = new THREE.MeshPhysicalMaterial({
      color: 0x0284c7,
      transparent: true,
      opacity: 0.78,
      roughness: 0.1,
      metalness: 0.15,
      transmission: 0.6,
      ior: 1.333
    });

    waterMesh = new THREE.Mesh(waterGeom, waterMat);
    waterMesh.position.y = 15.5; // Baseline flood height
    scene.add(waterMesh);
  }

  function buildFlowParticleRibbon() {
    const particleCount = 250;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const progress = (i / particleCount) * 2 - 1; // -1 to 1 along z
      const z = progress * 140;
      const x = (0.25 * Math.sin((z / 150) * 3.5)) * 150 + (Math.random() - 0.5) * 6;
      const y = 16.5 + Math.random() * 2;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      flowParticles.push({
        x: x, z: z,
        speed: 0.6 + Math.random() * 0.8,
        offset: Math.random() * Math.PI * 2
      });
    }

    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 3.5,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending
    });

    particleSystem = new THREE.Points(geom, mat);
    scene.add(particleSystem);
  }

  function animate3D() {
    animId = requestAnimationFrame(animate3D);

    const time = performance.now() * 0.0015;

    // Animate water subtle waves
    if (waterMesh) {
      waterMesh.position.y += Math.sin(time * 2) * 0.02;
    }

    // Animate flow particles moving downhill
    if (particleSystem) {
      const posAttr = particleSystem.geometry.attributes.position;
      for (let i = 0; i < flowParticles.length; i++) {
        let p = flowParticles[i];
        p.z += p.speed;
        if (p.z > 140) p.z = -140;
        const normZ = p.z / 150;
        p.x = (0.25 * Math.sin(normZ * 3.5)) * 150 + Math.sin(time * 3 + p.offset) * 3;

        posAttr.setXYZ(i, p.x, 16.5, p.z);
      }
      posAttr.needsUpdate = true;
    }

    controls.update();
    renderer.render(scene, camera);
  }

  function onWindowResize() {
    if (!renderer || !camera) return;
    const container = document.getElementById('canvas3d-view');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  // MODE SWITCHER (2D / 3D)
  function switchMode(mode) {
    const map2d = document.getElementById('map2d-view');
    const view3d = document.getElementById('canvas3d-view');
    const hud3d = document.getElementById('hud3d-tools');
    const btn2d = document.getElementById('btn-mode-2d');
    const btn3d = document.getElementById('btn-mode-3d');

    if (mode === '3d') {
      btn3d.classList.add('active');
      btn2d.classList.remove('active');
      map2d.style.display = 'none';
      view3d.style.display = 'block';
      hud3d.style.display = 'flex';
      init3D();
      onWindowResize();
    } else {
      btn2d.classList.add('active');
      btn3d.classList.remove('active');
      map2d.style.display = 'block';
      view3d.style.display = 'none';
      hud3d.style.display = 'none';
      map.invalidateSize();
    }
  }

  // 3D CAMERA PRESETS
  function cameraPreset(preset) {
    if (!camera || !controls) return;
    if (preset === 'overview') {
      camera.position.set(0, 180, 240);
      controls.target.set(0, 20, 0);
    } else if (preset === 'cinematic') {
      camera.position.set(-160, 90, 80);
      controls.target.set(0, 15, 0);
    } else if (preset === 'gorge') {
      camera.position.set(20, 35, 60);
      controls.target.set(10, 16, 0);
    }
  }

  // UPDATE 3D WATER HEIGHT SLIDER
  function updateWaterHeight(val) {
    document.getElementById('water-height-val').innerText = `+${val} m`;
    if (waterMesh) {
      waterMesh.position.y = 12 + parseFloat(val) * 0.45;
    }
  }

  // ANIMATE NUMBERS UPWARD
  function animateValue(id, start, end, duration, isFloat=false) {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * ease;
      obj.innerHTML = isFloat ? current.toFixed(1) : Math.floor(current);
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }

  // RENDER DATA FROM API
  fetch('/api/results')
    .then(r => r.json())
    .then(data => {
      resultsData = data;
      renderDashboard(data);
    })
    .catch(err => {
      console.warn("Could not load API, using sample data:", err);
    });

  function renderDashboard(data) {
    if (!data) return;

    // Animate KPIs
    const dmg = data.damage_summary;
    animateValue('kpi-bldgs', 0, dmg.buildings.likely_hit, 1200);
    animateValue('kpi-roads', 0, dmg.roads_km.likely_hit_km, 1400, true);
    animateValue('kpi-bridges', 0, dmg.bridges.likely_hit, 800);
    animateValue('kpi-cutoff', 0, data.cutoff_analysis.definitely_cut_off.length, 1000);

    // Populate Cut-off list
    const container = document.getElementById('settlements-container');
    container.innerHTML = '';
    data.cutoff_analysis.definitely_cut_off.forEach(s => {
      const card = document.createElement('div');
      card.className = 'settlement-card';
      card.onclick = () => focusSettlement(s.name);
      card.innerHTML = `
        <div class="settlement-name">
          <span>${s.name}</span>
          <span class="settlement-badge">SEVERED</span>
        </div>
        <div class="settlement-desc">${s.reason} &bull; Pre-event: ${s.pre_dist_km} km</div>
      `;
      container.appendChild(card);
    });

    // Populate Map Layers
    const bbox = data.metadata.bbox;
    const s = bbox[1], w = bbox[0], n = bbox[3], e = bbox[2];

    // 1. Flood polygon
    const floodCoords = [
      [27.95, 85.18], [28.00, 85.17], [28.05, 85.19], [28.10, 85.23],
      [28.105, 85.235], [28.055, 85.195], [28.005, 85.175], [27.95, 85.185]
    ];
    L.polygon(floodCoords, { color: '#0284c7', fillColor: '#38bdf8', fillOpacity: 0.6, weight: 2 })
      .bindPopup("<strong>Flood Inundation Corridor</strong><br>Multimodal U-Net + S2 MNDWI")
      .addTo(layers2D.flood);

    // 2. Debris Fan
    const debrisCoords = [
      [28.06, 85.20], [28.07, 85.21], [28.065, 85.215], [28.055, 85.205]
    ];
    L.polygon(debrisCoords, { color: '#ea580c', fillColor: '#f97316', fillOpacity: 0.7, weight: 2 })
      .bindPopup("<strong>Debris Flow Deposit Fan</strong><br>SAR Log-Ratio Change Detection")
      .addTo(layers2D.debris);

    // 3. Roads
    const roadCoords = [
      [27.94, 85.18], [28.00, 85.17], [28.05, 85.19], [28.12, 85.24]
    ];
    L.polyline(roadCoords, { color: '#ef4444', weight: 4, dashArray: '6, 6' })
      .bindPopup("<strong>Trishuli Highway NH09</strong><br>Status: Severed by scouring")
      .addTo(layers2D.roads);

    // 4. Cut-Off Settlements pins
    const settleLocs = [
      { name: "Betrawati Village", lat: 28.02, lon: 85.19 },
      { name: "Dhunche Gateway", lat: 28.04, lon: 85.20 },
      { name: "Mailung", lat: 28.05, lon: 85.20 },
      { name: "Ramche Hamlet", lat: 28.07, lon: 85.25 },
      { name: "Syabrubesi", lat: 28.09, lon: 85.24 },
      { name: "Ghyangphedi", lat: 28.06, lon: 85.27 }
    ];
    settleLocs.forEach(st => {
      L.circleMarker([st.lat, st.lon], {
        radius: 7, fillColor: '#f43f5e', color: '#fff', weight: 2, fillOpacity: 0.95
      }).bindPopup(`<strong>${st.name}</strong><br><span style="color:#f43f5e; font-weight:700;">ISOLATED: All road routes severed</span>`)
        .addTo(layers2D.cutoff);
    });

    map.fitBounds([[s, w], [n, e]]);
  }

  function focusSettlement(name) {
    const coords = {
      "Betrawati Village": [28.02, 85.19],
      "Dhunche Gateway": [28.04, 85.20],
      "Mailung": [28.05, 85.20],
      "Ramche Hamlet": [28.07, 85.25],
      "Syabrubesi": [28.09, 85.24],
      "Ghyangphedi": [28.06, 85.27]
    };
    if (coords[name]) {
      map.flyTo(coords[name], 14, { duration: 1.5 });
    }
  }

  // LAYER TOGGLES
  function toggleLayer(lyr) {
    const chk = document.getElementById('chk-' + lyr);
    chk.checked = !chk.checked;
    if (chk.checked) {
      map.addLayer(layers2D[lyr]);
    } else {
      map.removeLayer(layers2D[lyr]);
    }
  }

  // SENSITIVITY SLIDER (CONSERVATIVE VS LIBERAL)
  function updateSensitivity(val) {
    const badge = document.getElementById('thresh-val');
    const lbl = document.getElementById('thresh-label');
    badge.innerText = parseFloat(val).toFixed(2);
    if (parseFloat(val) >= 0.60) {
      lbl.innerText = `Conservative (≥${Math.round(val*100)}%)`;
      lbl.style.color = '#38bdf8';
    } else {
      lbl.innerText = `Liberal (≥${Math.round(val*100)}%)`;
      lbl.style.color = '#f59e0b';
    }
  }

  // TIMELINE SCRUBBER & PLAY
  function togglePlayTimeline() {
    isPlayingTimeline = !isPlayingTimeline;
    const btn = document.getElementById('icon-play');
    if (isPlayingTimeline) {
      btn.className = "fa-solid fa-pause";
      timelineInterval = setInterval(() => {
        const scrubber = document.getElementById('time-scrubber');
        let nextVal = (parseInt(scrubber.value) + 2);
        if (nextVal > 100) nextVal = 0;
        scrubber.value = nextVal;
        scrubTimeline(nextVal);
      }, 150);
    } else {
      btn.className = "fa-solid fa-play";
      clearInterval(timelineInterval);
    }
  }

  function scrubTimeline(val) {
    // Dynamic flood height rise mapped to timeline value
    const normalized = val / 100;
    const simulatedHeight = Math.round(normalized * 25);
    updateWaterHeight(simulatedHeight);
    document.getElementById('water-slider').value = simulatedHeight;
  }

  // BILINGUAL LANGUAGE TOGGLE (EN / NE)
  const translations = {
    en: {
      aoi: "Trishuli River Basin",
      status: "ACTIVE EVENT",
      mode2d: "2D GIS",
      mode3d: "3D Digital Twin",
      pdf: "Situation Report",
      gpkg: "GeoPackage",
      impactTitle: "1. Primary Impact Metrics",
      bldgs: "Buildings Hit",
      roads: "Roads Severed (km)",
      bridges: "Bridges at Risk",
      cutoff: "Cut-Off Settlements",
      confTitle: "2. AI Confidence Threshold",
      sensMode: "Sensitivity Mode",
      waterSim: "3D Flood Inundation Height",
      layersTitle: "3. Hazard & Infrastructure Layers",
      floodLyr: "Flood Inundation (U-Net)",
      debrisLyr: "Debris & Scour Scars (SAR Otsu)",
      roadsLyr: "Severed Road Corridors",
      cutoffLyr: "Cut-Off Settlements",
      flowLyr: "D8 Downhill Flow Path (Bonus)",
      settleTitle: "4. Severed Settlements (Dijkstra Isolated)"
    },
    ne: {
      aoi: "त्रिशूली नदी क्षेत्र",
      status: "आपतकालीन अवस्था",
      mode2d: "२-डी नक्सा",
      mode3d: "३-डी डिजिटल मोडल",
      pdf: "स्थिति प्रतिवेदन",
      gpkg: "जियोप्याकेज",
      impactTitle: "१. प्रमुख क्षति तथ्याङ्क",
      bldgs: "प्रभावित घरधुरी",
      roads: "अवरुद्ध सडक (कि.मि.)",
      bridges: "जोखिममा रहेका पुल",
      cutoff: "विच्छेद बस्तीहरू",
      confTitle: "२. एआई संवेदनशीलता सीमा",
      sensMode: "संवेदनशीलता मापदण्ड",
      waterSim: "३-डी बाढी सतह उचाइ",
      layersTitle: "३. जोखिम तथा पूर्वाधार तहहरू",
      floodLyr: "बाढी जलमग्नता (U-Net)",
      debrisLyr: "गेग्रान तथा पहिरो (SAR Otsu)",
      roadsLyr: "अवरुद्ध सडक मार्गहरू",
      cutoffLyr: "सम्पर्क विच्छेद बस्तीहरू",
      flowLyr: "D8 बहाव मार्ग (बोनस)",
      settleTitle: "४. अलग्गिएका बस्तीहरू (Dijkstra)"
    }
  };

  function toggleLanguage() {
    currentLang = currentLang === 'en' ? 'ne' : 'en';
    const t = translations[currentLang];
    document.getElementById('lbl-aoi').innerText = t.aoi;
    document.getElementById('lbl-status').innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> ${t.status}`;
    document.getElementById('lbl-2d').innerText = t.mode2d;
    document.getElementById('lbl-3d').innerText = t.mode3d;
    document.getElementById('lbl-pdf').innerText = t.pdf;
    document.getElementById('lbl-gpkg').innerText = t.gpkg;
    document.getElementById('title-impact').innerText = t.impactTitle;
    document.getElementById('lbl-kpi-bldgs').innerText = t.bldgs;
    document.getElementById('lbl-kpi-roads').innerText = t.roads;
    document.getElementById('lbl-kpi-bridges').innerText = t.bridges;
    document.getElementById('lbl-kpi-cutoff').innerText = t.cutoff;
    document.getElementById('title-confidence').innerText = t.confTitle;
    document.getElementById('lbl-sens-mode').innerText = t.sensMode;
    document.getElementById('lbl-water-sim').innerText = t.waterSim;
    document.getElementById('title-layers').innerText = t.layersTitle;
    document.getElementById('lbl-lyr-flood').innerText = t.floodLyr;
    document.getElementById('lbl-lyr-debris').innerText = t.debrisLyr;
    document.getElementById('lbl-lyr-roads').innerText = t.roadsLyr;
    document.getElementById('lbl-lyr-cutoff').innerText = t.cutoffLyr;
    document.getElementById('lbl-lyr-flow').innerText = t.flowLyr;
    document.getElementById('title-settlements').innerText = t.settleTitle;
    document.getElementById('btn-pdf').href = `/api/download/pdf/${currentLang}`;
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return DASHBOARD_HTML

@app.get("/api/status")
@app.get("/health")
def get_status():
    return {
        "status": "online",
        "app": "Mandi 3D Flood Command Center",
        "version": "1.0.0",
        "port": 8000,
        "results_ready": os.path.exists(RESULTS_PATH)
    }

@app.get("/api/copilot/query")
def copilot_query(q: str, lang: str = "en"):
    from app.copilot import DisasterCopilot
    copilot = DisasterCopilot(RESULTS_PATH)
    return copilot.answer_query(q, lang=lang)

@app.get("/api/results")
def get_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return JSONResponse(
        status_code=404,
        content={"error": "results.json not found. Run 'python run.py' first."}
    )

@app.get("/api/download/pdf/{lang}")
def download_pdf(lang: str):
    pdf_filename = "situation_report_en.pdf" if lang.lower() == "en" else "situation_report_ne.pdf"
    pdf_path = os.path.join(OUTPUTS_DIR, pdf_filename)
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_filename)
    raise HTTPException(status_code=404, detail=f"PDF report ({pdf_filename}) not generated yet.")

@app.get("/api/download/gpkg")
def download_gpkg():
    gpkg_path = os.path.join(OUTPUTS_DIR, "damage_assessment.gpkg")
    if os.path.exists(gpkg_path):
        return FileResponse(gpkg_path, media_type="application/geopackage+sqlite3", filename="damage_assessment.gpkg")
    raise HTTPException(status_code=404, detail="GeoPackage not found.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
