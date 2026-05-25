SELECT Churn, COUNT(*) AS TotalCustomers
FROM churn_data
GROUP BY Churn;

SELECT Contract, AVG(MonthlyCharges) AS AvgMonthlyCharge
FROM churn_data
GROUP BY Contract;

SELECT InternetService, COUNT(*) AS CustomerCount
FROM churn_data
GROUP BY InternetService;
