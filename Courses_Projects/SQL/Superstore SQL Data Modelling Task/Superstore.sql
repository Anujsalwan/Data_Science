CREATE DATABASE superstore;
USE superstore;

/*
2. Perform Below Data Retrieval Tasks
a. Top 5 Customers based on Sales.
b. Bottom 10 Customers based on Profits.
c. Running Sum of Sales of Corporate Segment from Past Three-Month Range
d. Moving Average of December Month from West Region
e. Retrieve the Data of Customers whose sales are greater than 10000 and who are from California.
f. Top 5 Products based on Profits with City of the Customers.
g. Retrieve the SUM of sales of each Product and Arrange the Sales in DESC Order
h. Retrieve the SUM of Profits of Each Customers and Arrange the Profits in DESC Order
*/

---a. Top 5 Customers based on Sales
select * from orders
 order by sales desc
 limit 5;
--- b. Bottom 10 Customers based on Profits.
select * from orders
 order by profit asc
 limit 10;
---c. Running Sum of Sales of Corporate Segment from Past Three-Month Range
SELECT 
    `Order Date`, 
    Sales,
    SUM(Sales) OVER (ORDER BY STR_TO_DATE(`Order Date`, '%d-%m-%Y')) AS Running_Total
FROM orders
WHERE Segment = 'Corporate';

---d. Moving Average of December Month from West Region
SELECT 
    `Order Date`, 
    Sales,
    AVG(Sales) OVER (ORDER BY STR_TO_DATE(`Order Date`, '%d-%m-%Y') ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS Moving_Avg
FROM orders
WHERE Region = 'West' 
AND MONTH(STR_TO_DATE(`Order Date`, '%d-%m-%Y')) = 12;

---e. Customers with Sales > 10,000 from California

SELECT `Customer Name`, SUM(Sales) AS Total_Sales

FROM orders

WHERE State = 'California'

GROUP BY `Customer Name`

HAVING Total_Sales > 10000; 


--- Verification Query for e.
SELECT `Customer Name`, SUM(Sales) AS Total_Sales
FROM orders
WHERE TRIM(State) = 'California'
GROUP BY `Customer Name`
order BY Total_Sales DESC;


---f. Top 5 Products based on Profits with City of the Customers.

SELECT `Product Name`, City, SUM(Profit) AS Total_Profit
FROM orders
GROUP BY `Product Name`, City
ORDER BY Total_Profit DESC
LIMIT 5;

---g. Sum of Sales per Product (Descending)

SELECT `Product Name`, SUM(Sales) AS Total_Sales
FROM orders
GROUP BY `Product Name`
ORDER BY Total_Sales DESC;

---h. Sum of Profits per Customer (Descending)

SELECT `Customer Name`, SUM(Profit) AS Total_Profit
FROM orders
GROUP BY `Customer Name`
ORDER BY Total_Profit DESC;