# Data Analytics Work Samples – Joyce Goh

This repository contains four representative analytics projects demonstrating my experience across statistical modeling, machine learning, geospatial analysis, large-scale data processing, and scientific computing. Each sample includes a summary of the problem, methodology, and results, along with code notebooks and visualizations.

---

## **1. Coral Geochemistry & NMDS Modeling (R)**

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

## **2. Large-Scale 6D Data Hunt with PySpark RDDs**

### **Overview**
This project processes and analyzes millions of 6-dimensional points using **PySpark RDDs**. This project was executed on the Bridges2 supercomputing cluster, using SSH to access compute nodes and run distributed Python workflows. Tasks include clustering, PCA projection, and computing cluster extents in projected space.

### **Key Techniques**
- PySpark RDD transformations (map, reduce, filter, zip)
- k-means clustering at scale
- Distributed PCA and projections
- Optimization of transformations and caching strategies

### **Files**
- `/pyspark_6d/` — PySpark script, write-up

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

## 4. SQL Queries on Customer Orders 

This folder contains two SQL queries that demonstrate my ability to write clear, correct SQL for analytics use cases, based on a data set of customer orders 

### Query 1 — Geospatial Customer Targeting

**File:** `queries.sql` (section: Geospatial zip ranking)

This query identifies the **top 3 postal codes by number of distinct orders** within 100 km of a reference location. It joins `Customer`, `Zips`, and `Orders`, computes geodesic distance using `ST_DISTANCE_SPHERE`, filters by radius, aggregates orders by postal code, and ranks the results.

### Query 2 — Running Total Revenue Over Time

**File:** `queries.sql` (section: Running total revenue)

This query calculates **daily revenue** and a **running total of revenue** across all orders using a window function. It groups orders by date, computes daily sums, and then applies `SUM(...) OVER (ORDER BY date)` to produce a cumulative revenue series.

**Key concepts:**
- Multi-table joins  
- Subqueries  
- Geospatial distance calculation  
- Date-based aggregation  
- Window functions  
- Running totals for time series analysis 
---

## Summary  
Across these three samples, this portfolio demonstrates:

- **Python, R, SQL**, and scientific computing strength  
- **Statistical modeling, causal reasoning, and data visualization**  
- Experience with **big data (PySpark)** and **HPC workflows**  
- Ability to extract actionable insights from complex datasets  
- Clear communication of technical work  

For any questions or additional samples, feel free to contact me.

