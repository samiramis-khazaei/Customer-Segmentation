# Customer Segmentation 
### Problem Statement:<br>
In today's dynamic business landscape, industries aim to attract new customers and retain their existing ones. Given the uniqueness of each customer, employing a one-size-fits-all approach is ineffective. Therefore, it is more effective for companies to categorize their customers based on their attributes and behaviors, aligning their marketing approaches closely with the characteristics of similar customer groups. Customer segmentation is the process of classifying customers based on shared behavior or attributes.
The objective of this project is to develop a customer segmentation model for a UK- based and registered non-store online retail business specializing in unique all-occasion gifts, with a significant portion of its customer base consisting of wholesalers. The model employs the RFM framework, assessing Recency, Frequency, and Monetary value as key indicators for analyzing customer behavior. These three dimensions aid in identifying distinct customer groups with similar characteristics.
### Code and Resources Used:<br>
**Jupiter Notebook**
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, worldcloud<br>
**Data:** https://archive.ics.uci.edu/dataset/352/online+retail<br>
### Data Collection & Cleaning:<br>
The dataset comprises 541,909 records and includes 8 features. The data collection period spans from December 1, 2011, to December 9, 2012. The features are :<br>
1 - InvoiceNo<br>
2 - StockCode<br>
3 - Description<br>
4 - Quantity<br>
5 - InvoiceDate<br>
6 - UnitPrice<br>
7 - CustomerID<br>
8 - Country<br>
The column "InvoiceNO" comprises a 6-digit integral number assigned uniquely to each transaction. If the code begins with the letter "C," it signifies a cancellation. The pie chart illustrates that 1.7% of the total purchases were canceled<br>
![Untitled](https://github.com/user-attachments/assets/98bdd558-ddfa-4eab-9a07-b11f5e3e2a5c)

The column "StockCode" contains product codes, each uniquely assigned to a distinct product. The "Description" column displays the product names. The word cloud illustrates the most frequent words in the "Description" column.<br>

![Untitled](https://github.com/user-attachments/assets/4e159abe-660e-4479-9ee9-acf95d03e13a)

The "Quantity" column represents the number of items purchased, "InvoiceDate" denotes the day and time of each transaction, and "UnitPrice" signifies the numeric product price per unit in sterling. The graphs below highlight specific instances where a substantial number of items were both purchased and canceled on the same day.<br>

![Untitled](https://github.com/user-attachments/assets/9d3b4762-0b8e-45b9-b146-68ca76af7fbc)

![Untitled-1](https://github.com/user-attachments/assets/96cdc7fd-83c3-4e2d-b0dc-7f7608993092)

These unusual figures are depicted in the box plot below, revealing outliers within the data frame. Additionally, it is noteworthy that there are instances of negative quantities and zero prices for certain items.<br>

![Untitled](https://github.com/user-attachments/assets/58bff88f-1a1b-44e7-ae4b-723d5357a5ce)

The "Country" column specifies the name of the country where each customer resides, while the "CustomerID" column consists of unique 5-digit integral numbers assigned to individual customers. The map illustrates that the highest amount of money was spent by customers in the UK.<br>

![Untitled](https://github.com/user-attachments/assets/715003e9-0527-4a53-a2ca-79f09360d1ef)

After analyzing the data frame, I performed the following data-cleaning steps:<br>
• Handling Missing Values: The "CustomerID" column had 135,080 missing values. While this is about 24% of the data, the customer evaluation relies on this identifier, therefor I deleted the rows with null values, as there was no suitable replacement.<br>
• Removing Specific Transactions: Since the company deals with wholesale customers, it is not unusual for customers to purchase a large number of items. Therefore, data cleaning is a lengthy process, and only items that are both purchased and refunded on the same day are deleted. For example, on 2011/01/08, 74,215 items were purchased and returned. Similarly, on 2011/12/09, 80,995 items were purchased and returned. These transactions were deleted.<br>
• Eliminating Items with Zero Price or Quantity: Items with zero prices or quantities were identified and removed from the dataset.<br>
These data-cleaning measures were implemented to enhance the quality and reliability of the dataset.<br>

### Feature Engineering:<br>
To undertake machine learning, the initial step involves defining key metrics. Three crucial indicators—recency, frequency, and monetary value—are employed to analyze customer behavior and identify customer groups with shared characteristics.<br>
Understanding RFM (Recency, Frequency, Monetary) analysis provides valuable insights into current customer engagement levels (recency), identifies loyal and regular customers (frequency), and highlights high-value customers (monetary value).<br>
- Monetary Value: Monetary value helps spotlight customers who contribute the most revenue to your business. It represents the total amount of money a specific person has spent with the company.<br>
![Untitled](https://github.com/user-attachments/assets/1fd7353a-8bca-42bf-bf3b-9bb3a05e7327)

- Frequency: Frequency reveals which customers are loyal and consistently engage with the company by measuring their interaction frequency within a specific timeframe.<br>
![Untitled](https://github.com/user-attachments/assets/c8066b53-089c-42f3-a8b5-4a6e4972678b)

- Recency: Recency provides insights into current customer engagement levels by measuring the time elapsed since their last purchase.
![Untitled](https://github.com/user-attachments/assets/ce767ca1-f105-4cd6-a524-9ff54c02cd2a)

Understanding these metrics empowers businesses to make informed decisions about optimizing marketing strategies, increasing retention, identifying top customers, and ultimately driving business success.<br>
After establishing the key metrics—monetary value, recency, and frequency—their respective mean values of 1900.98, 91.06, and 93.08 reveal substantial differences in magnitudes. Neglecting to standardize these disparate scales can introduce bias, especially in machine learning algorithms sensitive to feature magnitudes, such as k-means clustering. Standardization becomes pivotal to ensuring each feature contributes proportionally to the analysis, avoiding dominance by those with larger scales. Moreover, it aids algorithm convergence by enhancing optimization speed during model training—a critical consideration for iterative optimization processes. Despite standardization, the data retains a skewed distribution. To address this, I applied logarithmic transformations, particularly effective for right-skewed distributions where data is concentrated on the left with a long tail to the right. This transformation compresses larger values, promoting a more symmetric distribution. The resulting 3D scatter plot, post-standardization, and logarithmic transformation is presented<br>

![Untitled](https://github.com/user-attachments/assets/12b29b86-54a7-4fb3-a37c-bdc2204b854d)

### Model Building:<br>
**K-means clustering**<br>
**Gaussian Mixture clustering**<br>
To assess the performance of clustering algorithms, I employed the Silhouette Score, a metric that gauges how similar an object is to its own cluster compared to others. A higher silhouette score signifies more well-defined clusters. The evaluation yielded a silhouette score of 0.34 for K-means clustering and 0.3 for Gaussian Mixture clustering. Consequently, the K-means clustering algorithm was selected, as it demonstrated a higher silhouette score, indicating better-defined clusters compared to the Gaussian Mixture clustering algorithm.<br>

![Untitled](https://github.com/user-attachments/assets/1ab3fe09-4d08-4a0d-82dc-e7f35536d17a)

![Untitled-1](https://github.com/user-attachments/assets/bec41b0a-2c17-4b1d-908e-3288b083646b)

### Outcome:<br>
The table summarizes the outcome of the K-means clustering. The presented table displays the mean values of the three features for each cluster, offering valuable insights for the company to tailor distinct marketing strategies. For instance, analysis of the table reveals that Cluster 0 is characterized by customers who make substantial transactions (highest monetary value), haven't made recent purchases (lowest recency), yet exhibit high engagement and loyalty through frequent transactions (highest frequency). This information can inform targeted marketing strategies aimed at retaining and maximizing the value of this customer segment.<br>
In contrast, Cluster 1 appears to represent a segment of customers with different characteristics:<br>
• They make smaller transactions (lowest monetary value).<br>
• They have made recent purchases (highest recency).<br>
• Their engagement with the business is less frequent (lowest frequency).<br>
In summary, the clustering results provide actionable insights, allowing the company to implement more effective and targeted marketing strategies tailored to the specific needs and behaviors of different customer segments.<br>

![image](https://github.com/user-attachments/assets/c6b29c68-dd56-4491-a9c5-2d02cf99a3ec)





























