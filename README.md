

##  **\*\*Project: Predicting House Prices with Random Forest \& XGBoost\*\***

#### Abstract



This project explores the application of machine learning models to predict housing prices in Melbourne using structured real estate data.

By leveraging Random Forest Regressor and XGBoost Regressor, it evaluates the predictive influence of property features such as total rooms, bathrooms, land size, and geographic coordinates.

Through extensive exploratory data analysis (EDA), feature importance assessment, and performance evaluation using R², MAE, and RMSE, this study identifies the key drivers of property value and visualizes spatial price distributions across the city.

The project integrates statistical modeling and geospatial visualization to provide actionable insights for urban development, property valuation, and real estate investment.



#### Overview

This project explores how two advanced machine learning models that is\*\*Random Forest Regressor\*\* and \*\*XGBoost Regressor\*\* ,can predict house prices accurately. It compares their performance to determine which property and location factors most influence prices.



The analysis includes visual tools like error metrics, scatter plots, and feature importance graphs to show model accuracy. 

An \*\*interactive map\*\* also highlights price patterns across regions, offering clear insights into how location affects housing costs.



#### Objectives



\-Build and train Random Forest and XGBoost models for housing price prediction.



\-Evaluate their performance using R², MAE, and RMSE metrics.



-Identify which features most influence house prices.



-Visualize findings using correlation heatmaps, scatter plots, and an interactive geospatial map.



#### Models Used



| Model                       | Description                                                                      | Strengths                                                  |

| --------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------- |

| \*\*Random Forest Regressor\*\* | Ensemble of multiple decision trees averaging predictions to reduce overfitting. | Robust, stable, and interpretable.                         |

| \*\*XGBoost Regressor\*\*       | Gradient boosting framework that corrects errors iteratively.                    | Highly efficient and powerful for structured/tabular data. |



#### Evaluation Metrics

| Metric                             | Description                                                        | Interpretation                           |

| ---------------------------------- | ------------------------------------------------------------------ | ---------------------------------------- |

| \*\*R² (R-squared)\*\*                 | Proportion of variance in prices explained by the model.           | Closer to 1 = stronger predictive power. |

| \*\*MAE (Mean Absolute Error)\*\*      | Average absolute difference between predictions and actual values. | Lower = higher accuracy.                 |

| \*\*RMSE (Root Mean Squared Error)\*\* | Penalizes large errors more heavily.                               | Lower = fewer large deviations.          |



#### Model Evaluation Results

| Model             | R² Score | MAE (AUD)  | RMSE (AUD) |

| ----------------- | -------- | -------- | -------- |

| \*\*Random Forest\*\* | 0.805    | AUD159,254 | AUD242,034 |

| \*\*XGBoost\*\*       | 0.813    | AUD158,309 | AUD237,159 |



##### Interpretation:


![MAE vs RMSE Comparison](/Model%20Evaluation%20comparison%20chart.png)



Both models perform strongly, explaining over 80% of the variance in house prices.



XGBoost slightly outperforms Random Forest, with lower error metrics and higher R².



The small performance gap shows that both models are well-tuned and reliable.
### Price distribution in Melbourne
![Price distribution histogram](/histogram%20of%20price%20distribution.png)

 **Insights from the Price Distribution**


| Observation              | Meaning                                          |
| ------------------------ | ------------------------------------------------ |
| Right-skewed             | Few very expensive homes skewing the mean upward |
| Peak around 0.5–0.9M AUD | Typical house price range                        |
| Long tail beyond 1.5M    | Luxury or high-demand zone properties            |
| Skewness                 | Log-transform useful before regression           |
| Market implication       | Two-tier market: mass mid-range vs luxury        |


#### Feature Importance Comparison

##### 

##### Insights:
![Feature Importance Comparison](/Feature%20Importance%20Comparison%20chart.png)
![Correlation Heat Map](/correlation%20heatmap.png)

\-total\_rooms is the most influential predictor across both models, indicating that overall property size strongly impacts price.



\-Geolocation features (latitude, longitude) are also critical, confirming that neighborhood and accessibility matter significantly.



\-bathroom, landsize, and buildingarea add secondary but meaningful influence.



\-Features like age and car have smaller contributions but may capture specific urban or suburban patterns.

### **Interpretation**

\-Overall, the heatmap confirms that **house size, number of rooms, and amenities** are the most influential predictors of house prices in Melbourne, while **location and age** have secondary but meaningful effects.

\-The **high alignment between predicted and actual prices** indicates a strong model fit.

\-**NB** :since there are some redundant features like `rooms` and `bedroom2`, I avoid them to prevent multicollinearity during modeling. I  instead use the `total_rooms` which combines them. 



#### Actual vs Predicted Prices



##### Interpretation:



1. The scatter plot shows actual vs predicted prices for both models.
   
2. The clustering of points along the diagonal line confirms high predictive accuracy.
   
3. XGBoost’s tighter clustering suggests it slightly outperforms Random Forest in precision.
   
4. Outliers on the far edges may represent luxury or atypical properties outside normal pricing patterns.



#### Interactive Map Visualization



The interactive price map displays geographic price variation across Melbourne using Folium.

##### 

##### Key Features:


* Each property is represented by a marker whose color and size correspond to predicted prices.
* High-priced homes are concentrated in specific inner-city or coastal regions.
* Hovering over points reveals property attributes such as room count and building area.
* Provides an intuitive spatial understanding of price clusters and regional disparities.



File: melbourne\_predicted\_house\_prices\_map.html

To open the map locally:  open melbourne\_predicted\_house\_prices\_map.html

#### 

#### Technologies Used



* Python 3.x
* Pandas, NumPy – Data cleaning and manipulation
* Scikit-learn – Random Forest, metrics, preprocessing
* XGBoost – Gradient boosting implementation
* Matplotlib, Seaborn – Visualization and EDA
* Folium – Interactive mapping and spatial analysis



#### How to Run the Project



1. ###### Clone the repository



git clone https://github.com/yourusername/house-price-prediction.git

cd house-price-prediction





###### 2. Install dependencies



pip install -r requirements.txt





###### 3. Run the Jupyter Notebook



jupyter notebook melbourne\_house\_prediction\_analysis.ipynb





###### 4. View the interactive map



open house\_prices\_map.html



#### Future Improvements



* Perform hyperparameter tuning using GridSearchCV or Optuna.



* Experiment with LightGBM or CatBoost for performance benchmarking.



* Engineer additional features (e.g., distance to city center, neighborhood category).



* Integrate temporal price data for time-series forecasting.



* Deploy a Streamlit web app for user-interactive house price predictions.



#### License



This project is released under the MIT License.



#### Acknowledgments



This project was inspired by real-world applications of machine learning in real estate analytics.

It combines predictive modeling with geospatial insight, providing valuable perspective for data scientists, property analysts, and urban planners seeking to understand housing market behavior.

