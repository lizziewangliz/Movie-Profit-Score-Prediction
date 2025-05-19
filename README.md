# Movie Profit and Score Prediction

## Overview

This project was developed as part of a data science team assignment to assist film producers in making informed, data-driven decisions during pre-production. Using a dataset of past films containing metadata such as director, budget, genre, and cast, we built and evaluated predictive models for two key outcomes:

- **Profit**: Estimated gross revenue minus budget
- **IMDb Score**: A proxy for audience reception

Our modeling framework enables producers to test different configurations (e.g., casting, genre, or production budget) and receive estimated financial and critical outcomes based on historical data.

---

## Business Context

In the film industry, uninformed decisions about casting, crew, and budgeting can lead to box office flops. This project aims to support **producers** by identifying high-impact predictors of profitability and critical success, allowing them to reduce risk and maximize returns through predictive modeling.

---

## Dataset Summary

- **Source**: IMDb (public scraped data)
- **Size**: ~3,000 films with attributes including:
  - Budget, Gross Revenue, Profit (engineered)
  - Director, Writer, Star, Genre, Company, Country
  - Audience Score, Number of Votes, Runtime, Rating

- **Target Variables**:
  - `profit` = gross - budget
  - `score` = IMDb user rating (1–10 scale)

---

## Modeling Techniques

We applied three modeling techniques with 5-fold cross-validation:

### 1. Random Forest (Best Performance)
- Handles nonlinear interactions and high-cardinality variables
- Highest R² and lowest RMSE for both profit and score predictions

### 2. K-Nearest Neighbors (KNN)
- Simple, interpretable baseline
- Best `k` = 5 (profit), `k` = 11 (score)

### 3. Post-Lasso Linear Regression
- Feature selection via Lasso regularization
- Retains interpretability while reducing dimensionality

---

## Evaluation Metrics

Each model was evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Out-of-sample R-squared (R²)

Results showed that **Random Forest consistently outperformed** KNN and Linear Regression for both targets.

---

## Key Insights

- **Profitability and score are influenced by cast, director, genre, and country**
- **Callout Features**: Budget, star power, and company consistently appeared as top predictors
- **Simplification**: Only the top 10 categories for directors, stars, writers, and companies were retained to reduce noise
- **Bias risks**: Model performs better on high-budget, mainstream films; caution is advised when evaluating indie projects

---

## Deployment Use Cases

- **Scenario Simulation**: Test multiple casting or budget configurations to compare expected outcomes
- **Targeted Production Planning**: Use model insights to adjust genre or location for optimal market performance
- **Audience Segmentation**: Predict how specific segments might respond to genre/star combinations

---

## Files

- `Team Project_Team 27_R.R` – Full R script including data prep, modeling, and evaluation
- `DS_Team Project_Team27.pdf` – Written project report
- `DS_Team Project_Team27.pptx` – Slide deck with visual summaries

---

## How to Run

1. Clone this repository.
2. Make sure `movies.csv` is placed in the working directory.
3. Open and run `Team Project_Team 27_R.R` in RStudio.
4. Required packages (install if needed):

```r
install.packages(c("caret", "randomForest", "glmnet", "MatchIt", "ggplot2", "reshape2", "class"))
