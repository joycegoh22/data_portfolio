# Split wb_data into 3 separate tables for easier queries -------

CREATE TABLE dim_country AS
SELECT DISTINCT country_code, country_name
FROM wb_data;

CREATE TABLE dim_series AS
SELECT DISTINCT series_code, series_name
FROM wb_data;

ALTER TABLE wb_data
MODIFY series_code VARCHAR(50), 
MODIFY country_code VARCHAR(50);

ALTER TABLE dim_series
MODIFY series_code VARCHAR(50);

ALTER TABLE dim_country
MODIFY country_code VARCHAR(50);

ALTER TABLE dim_country ADD PRIMARY KEY (country_code);
ALTER TABLE dim_series  ADD PRIMARY KEY (series_code);

# Check that tables are correctly split 
SELECT COUNT(DISTINCT country_code) FROM wb_data;
SELECT COUNT(*) FROM dim_country;

SELECT COUNT(DISTINCT series_code) FROM wb_data;
SELECT COUNT(*) FROM dim_series;

SELECT DISTINCT dim_series.series_code,
		series_name
FROM dim_series
JOIN wb_data 
ON dim_series.series_code = wb_data.series_code;

# Add which SDG goal each series is from 
ALTER TABLE dim_series
ADD COLUMN sdg_goal VARCHAR(20);

UPDATE dim_series
SET sdg_goal = CASE
    WHEN series_code = 'EG.EGY.PRIM.PP.KD' THEN 'SDG 7'   -- Energy intensity 
    WHEN series_code = 'EG.FEC.RNEW.ZS' THEN 'SDG 7' -- Renewable energy consumption
    WHEN series_code = 'EN.ATM.CO2E.PC' THEN 'SDG 9' -- CO2 emissions
    WHEN series_code = 'EN.ATM.PM25.MC.M3' THEN 'SDG 11'  -- PM2.5
    WHEN series_code = 'GB.XPD.RSDV.GD.ZS' THEN 'SDG 9' -- R&D
    WHEN series_code = 'NY.ADJ.SVNX.GN.ZS' THEN 'SDG 12' -- Adjusted net savings
    ELSE 'Other'
END
WHERE series_code IN (
    'EG.EGY.PRIM.PP.KD',
    'EG.FEC.RNEW.ZS',
    'EN.ATM.CO2E.PC',
    'EN.ATM.PM25.MC.M3',
    'GB.XPD.RSDV.GD.ZS',
    'NY.ADJ.SVNX.GN.ZS'
);

# Check join
SELECT w.year, c.country_name, s.series_name, w.value
FROM wb_data w
JOIN dim_country c USING (country_code)
JOIN dim_series s USING (series_code)
LIMIT 10;

# Drop country_name and series_name from wb_data
ALTER TABLE wb_data
DROP COLUMN country_name,
DROP COLUMN series_name;

# Set keys for wb_data
ALTER TABLE wb_data
ADD UNIQUE KEY uq_obs (country_code, series_code, year);


# Singapore's progress based on 2030 Green Plan pillars --------
SELECT year, 
        series_name,
		value,
        ROUND(value - LAG(value) OVER (PARTITION BY w.series_code ORDER BY year), 4) AS yoy_change,
        ROUND((value - LAG(value) OVER (PARTITION BY w.series_code ORDER BY year)) / LAG(value) OVER (PARTITION BY w.series_code ORDER BY year), 4) AS yoy_pct_change
FROM wb_data w
JOIN dim_series d
ON w.series_code = d.series_code
WHERE country_code = 'SGP'
ORDER BY w.series_code, year;


# Average annual change by pillar 
WITH sg AS (
  SELECT
    w.country_code,
    w.series_code,
    d.sdg_goal,
    d.series_name,
    w.year,
    w.value
  FROM wb_data w
  JOIN dim_series d
    ON w.series_code = d.series_code
  WHERE w.country_code = 'SGP'
    AND w.value IS NOT NULL
),

bounds AS (
  SELECT
    series_code,
    MIN(year) AS min_year,
    MAX(year) AS max_year
  FROM sg
  GROUP BY series_code
),

endpoints AS (
  SELECT
    s.series_code,
    b.min_year,
    b.max_year,
    MAX(CASE WHEN s.year = b.min_year THEN s.value END) AS min_year_value,
    MAX(CASE WHEN s.year = b.max_year THEN s.value END) AS max_year_value
  FROM sg s
  JOIN bounds b
    ON s.series_code = b.series_code
  GROUP BY s.series_code, b.min_year, b.max_year
),

yoy AS (
  SELECT
    s.series_code,
    s.sdg_goal,
    s.series_name,
    s.year,
    s.value,
    (s.value - LAG(s.value) OVER (PARTITION BY s.series_code ORDER BY s.year)) AS yoy_change,
    (s.value - LAG(s.value) OVER (PARTITION BY s.series_code ORDER BY s.year))
      / NULLIF(LAG(s.value) OVER (PARTITION BY s.series_code ORDER BY s.year), 0) AS yoy_pct_change
  FROM sg s
)

SELECT
  y.sdg_goal,
  y.series_name,
  (e.max_year - e.min_year + 1) AS n_years_observed,
  ROUND(AVG(y.yoy_pct_change), 3) AS avg_yoy_pct_change,
  ROUND((e.max_year_value - e.min_year_value) / NULLIF((e.max_year - e.min_year), 0), 4) AS avg_change_per_year
FROM yoy y
JOIN endpoints e
  ON y.series_code = e.series_code
GROUP BY y.sdg_goal, y.series_name, e.min_year, e.max_year, e.min_year_value, e.max_year_value
ORDER BY y.sdg_goal, y.series_name;


# Comparison with peers ----------------

# Comparing average values across the years 
SELECT sdg_goal,
		series_name,
		ROUND(AVG(CASE WHEN country_code = 'SGP' THEN value END), 2) AS sgp_avg,
        ROUND(AVG(CASE WHEN country_code = 'USA' THEN value END), 2) AS usa_avg,
        ROUND(AVG(CASE WHEN country_code = 'GBR' THEN value END), 2) AS uk_avg,
        ROUND(AVG(CASE WHEN country_code = 'FIN' THEN value END), 2) AS fin_avg,
        ROUND(AVG(CASE WHEN country_code = 'AUS' THEN value END), 2) AS aus_avg,
        ROUND(AVG(CASE WHEN country_code = 'JPN' THEN value END), 2) AS jpn_avg,
        ROUND(AVG(CASE WHEN country_code = 'PRK' THEN value END), 2) AS sk_avg,
        ROUND(AVG(CASE WHEN country_code = 'HKG' THEN value END), 2) AS hk_avg
FROM wb_data w
JOIN dim_series d
ON w.series_code = d.series_code
GROUP BY sdg_goal, series_name;

# Comparing yoy change
SELECT year, 
        series_name,
		value,
        country_code,
        ROUND(value - LAG(value) OVER (PARTITION BY w.series_code, country_code ORDER BY year), 4) AS yoy_change,
        ROUND((value - LAG(value) OVER (PARTITION BY w.series_code, country_code ORDER BY year)) / LAG(value) OVER (PARTITION BY w.series_code, country_code ORDER BY year), 4) AS yoy_pct_change
FROM wb_data w
JOIN dim_series d
ON w.series_code = d.series_code
ORDER BY w.series_code, year;

# Ranking countries every year 
SELECT year, 
		series_name, 
        country_code, 
        value,
        DENSE_RANK() OVER (PARTITION BY w.series_code, year ORDER BY (CASE WHEN series_name IN ('Renewable energy consumption', 'R&D', 'Adjusted net savings') THEN value ELSE -value END) DESC) AS country_rank
FROM wb_data w
JOIN dim_series d
ON w.series_code = d.series_code 
WHERE value IS NOT NULL;

        