-- [Query 1] Geospatial Customer Targeting 

SELECT sub.PostalCode AS zip, 
       COUNT(DISTINCT o.OrderID) AS no_of_orders
FROM Orders AS o
JOIN (SELECT c.CustomerID, z.Lng, z.Lat, c.PostalCode,
              ST_DISTANCE_SPHERE(POINT(z.Lng, z.Lat), POINT(-71.10253, 42.36224))/1000 AS distance
      FROM Customer AS c
      JOIN Zips AS z
      ON c.PostalCode = z.zip) AS sub 
ON o.CustomerId = sub.CustomerID
WHERE sub.distance <= 100
GROUP BY sub.PostalCode
ORDER BY no_of_orders DESC 
LIMIT 3;


-- [Query 2] Daily revenue and running total revenue across all orders

SELECT
    DATE(o.OrderDate) AS order_date,
    SUM(o.OrderAmount) AS daily_revenue,
    SUM(SUM(o.OrderAmount)) OVER (
        ORDER BY DATE(o.OrderDate)
    ) AS running_total_revenue
FROM Orders AS o
GROUP BY DATE(o.OrderDate)
ORDER BY order_date;
