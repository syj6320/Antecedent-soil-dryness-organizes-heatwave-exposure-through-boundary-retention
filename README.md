# Antecedent-soil-dryness-organizes-heatwave-exposure-through-boundary-retention
Code and workflows
Code repository for the manuscript:

**Antecedent soil dryness organizes heatwave exposure through boundary retention**

This repository contains the preprocessing, event-object extraction, mechanism diagnostics, robustness analysis, CMIP6 projection analysis, and publication-figure workflows used in the manuscript. The central aim is to test whether antecedent soil dryness organizes heatwave exposure primarily through the spatial retention and boundary evolution of heatwave objects, rather than only through changes in local heatwave intensity or duration.

---

## Repository description for GitHub

Event-object analysis code for studying how antecedent soil dryness organizes heatwave exposure through boundary retention. The repository includes ERA5/ERA5-Land preprocessing, 3D CC3D heatwave-object extraction, C26/C18/C6 connectivity sensitivity tests, front-versus-interior transition diagnostics, circulation and surface-energy mechanism figures, CMIP6 SSP/MME projections, and publication-ready figure workflows for the manuscript.

---

## 1. Scientific framework

The workflow is organized around an event-object view of heatwaves.

Instead of treating heatwaves only as isolated grid-cell exceedances, the analysis tracks three-dimensional connected heatwave objects in longitude–latitude–time space. Each object is then linked to antecedent or concurrent soil-moisture state, surface-energy partitioning, circulation controls, advancing-front recruitment, and interior/local retention.

The central hypothesis is:

> Antecedent soil dryness organizes heatwave exposure by stabilizing and retaining dry-state heatwave-object boundaries, especially at advancing fronts, thereby increasing object-scale exposure even when circulation and event geometry are controlled.

The core analysis therefore combines:

1. **Heatwave-object extraction**  
   Three-dimensional connected-component tracking of summertime heatwave grid cells.

2. **Soil-moisture state classification**  
   Six local soil-moisture states, S1–S6, from driest to wettest.

3. **Front/local decomposition**  
   Separation of retained interior/local cells from newly recruited advancing-front cells.

4. **Mechanism diagnostics**  
   Circulation, surface-energy partitioning, land-surface coupling, thermal advection, moisture divergence, and event geometry.

5. **Robustness and projections**  
   Connectivity sensitivity, threshold sensitivity, resolution sensitivity, lifecycle representativeness, and CMIP6 future projections.
