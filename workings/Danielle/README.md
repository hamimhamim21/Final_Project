# Predicting house prices in Melbourne - Final Project

For this project, we have used Machine Learning to predict house prices in Melbourne and in order to assess whether or not current house listings are over or undervalued. We have combined this with the Victorian crime rates as this would provide further insight in whether crime rates affect house prices. This will help a potential house buyer whether or not to bid on a property. 

- Data sources
- Libraries
- Overview repository
- Steps taken
- Caveats
- Potential improvements

The project team consists of:
Catherine Sloan
Danielle Cahill
Anne Wieggers
Tom Peddlesden
Joe Quinn

## Data sources 
We have used the following datasources:
- Training data:
[Melbourne house prices](https://www.kaggle.com/anthonypino/melbourne-housing-market?select=Melbourne_housing_FULL.csv)

- Testing data: 
[Domain](https://www.domain.com.au/)

We have supplemented the house price data with the Victorian crime rates, which can be found [here](https://discover.data.vic.gov.au/dataset/crime-by-location-data-table) .


## Libraries and technologies used

- Machine Learning
	- Sci kit learn
	- Joblib

- Back end
	- Flask
	- Flask PyMongo
	- Numpy	
	- scrape_mars
	- requests
	- BeautifulSoup
	- OS
	- Pandas
	- Splinter
	- ChromeDriverManager
	- Time
	- Sys

- Front end
	- HTML
	- CSS
	- Javascript


## Overview repository

## Steps

1. Retrieving training dataset
- Data cleaning (drop non relevant columns, null values, merging dataframes)

2. Retrieving testing data through scraping 
- Retrieve page with chrome driver manager and get URL
- Parse URL into beautiful soup to scrape data
- Initialise browser / return house features

3. Create Flask app to create routes to the relevant data points used
- Creating routes for index.html, tables/visualisations (if any?)
	- Route to render index.html
	- Route to trigger the scrape function (parse in Domain URL and scrape info from website)

4. Setting up model 
- multiple linear regression model using sklearn
	- read in csvs
	- correlation matrix
	- assign data to x and y values
	- convert categorial data (suburbs) to numbers through one-hot encoding/get_dummies
	- scaling and normalisation
	- fit the Model to the scaled training data and make predictions using the scaled test data

5. Prepare web page 
-	Page with predictive data
-	Page with data sources

6. Heroku deployment

Caveats
-	Dropped columns that had nulls/NAs. 
-	Bedrooms 2 came from other source to ignored and used number of rooms instead
-	What date to scrape and pass in to the model for the prediction. Not all listings are auctions so we don’t know the date that they will sell. Using the date the prediction is made instead. 
-	Use bootstrap templates / include reference/credit on webpage


## Potential improvements