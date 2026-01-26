# Data Analytics Work Samples – Joyce Goh

This repository contains four representative analytics projects demonstrating my experience across data analysis, data visualization, and statistical modelling. Each sample includes a summary of the problem, methodology, and results, along with code notebooks and visualizations.

---

## **1. Benchmarking Singapore Against Its Peers Using the Singapore Green Plan 2030 **

### Overview
This project benchmarks Singapore’s sustainability performance against peer economies using **6 SDG-aligned indicators** mapped to the **Singapore Green Plan 2030**. It evaluates both:
- **Singapore’s time trends** (progress over time), and
- **relative performance** versus peers (rank/positioning by year and indicator).

### Key Techniques
- **Python (pandas, numpy):** data cleaning, reshaping, validation
- **SQL:** joins, running totals, YoY absolute and % changes, ranking logic
- **Tableau:** interactive dashboarding and storytelling visualizations

### **Files**
- `/sg_greenplan2030/` — Python & SQL scripts, Tableau workbook, write-up

-- 

## **2. Coral Geochemistry & NMDS Modeling (R)**

### **Overview**
This project analyzes geochemical trace-element ratios across multiple coral species exposed to different pCO₂ levels. The objective is to understand species-specific geochemical responses to ocean acidification and identify multivariate patterns in coral chemistry.

### **Key Techniques**
- Non-metric Multidimensional Scaling (NMDS)
- PCA and ordination-based visualization
- Regression modeling and confidence ellipses
- Data wrangling and visualization in **R** (vegan, ggplot2)

### **Files**
- `/coral_nmds/` — R script, write-up

---

## **3. Hokkaido Earthquake Slip Partitioning Analysis (R, Python & GMT)**

### **Overview**
This geophysical analysis examines slip-partitioned regions in the Hokkaido trench using focal mechanism catalogs and GPS velocity fields. The objective is to understand deformation patterns and identify underlying tectonic structures.

### **Key Techniques**
- Processing earthquake catalogs (depth, strike, dip, rake filtering)
- Vector geometry of slip and trench-parallel vs trench-perpendicular components
- Visualization using **GMT**, ggplot2, and geospatial tools
- Interpretation of geophysical patterns and regional tectonics

### **Files**
- `/hokkaido_slip/` — R script, write-up

---

## **4. Large-Scale 6D Data Hunt with PySpark RDDs**

### **Overview**
This project processes and analyzes millions of 6-dimensional points using **PySpark RDDs**. This project was executed on the Bridges2 supercomputing cluster, using SSH to access compute nodes and run distributed Python workflows. Tasks include clustering, PCA projection, and computing cluster extents in projected space.

### **Key Techniques**
- PySpark RDD transformations (map, reduce, filter, zip)
- k-means clustering at scale
- Distributed PCA and projections
- Optimization of transformations and caching strategies

### **Files**
- `/pyspark_6d/` — PySpark script, write-up

-- 

## Summary  
Across these three samples, this portfolio demonstrates:

- **Python, R, SQL**, and scientific computing strength  
- **Statistical modeling, causal reasoning, and data visualization**  
- Experience with **big data (PySpark)** and **HPC workflows**  
- Ability to extract actionable insights from complex datasets  
- Clear communication of technical work  

For any questions or additional samples, feel free to contact me.

