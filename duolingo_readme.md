# Predicting Word Recall Probability in Duolingo using Apache Spark

## Project Overview

This project explores how machine learning can be used to model human
memory in large-scale language learning platforms. Using the Duolingo
Spaced Repetition dataset, we built a machine learning pipeline in
Apache Spark to predict the probability that a user will recall a word
correctly during a learning session.

The dataset contains over **13 million user-word interaction records**,
making it a realistic big data problem. To handle this scale
efficiently, the project was implemented using **Apache Spark on
Databricks Community Edition**, allowing distributed data processing and
scalable machine learning.

The final model achieved an **R² score of 0.82 using Random Forest
Regression**, demonstrating strong predictive performance for user
recall behavior.

------------------------------------------------------------------------

## Problem Statement

Language learning platforms such as Duolingo rely on **spaced repetition
systems** to decide when users should review vocabulary.

The key question is:

**What is the probability that a learner will recall a word at a given
moment?**

Accurately predicting recall probability helps learning platforms: -
personalize practice sessions - optimize review timing - improve
long‑term retention

This project frames the task as a **regression problem**, predicting the
variable **p_recall**, which represents the probability of correct
recall.

------------------------------------------------------------------------

## Dataset

We used the **Duolingo SLAM dataset**, which contains more than **13
million interaction records** from language learners.

Each row represents a user interacting with a vocabulary word.

### Key Features

  Feature             Description
  ------------------- ----------------------------------------------------
  p_recall            Target variable -- probability of recalling a word
  delta               Time since last exposure to the word
  user_id             Anonymized user identifier
  learning_language   Language the user is learning
  ui_language         Interface language
  lexeme_string       Vocabulary word
  history_seen        Number of previous exposures
  history_correct     Number of correct historical answers
  session_seen        Number of times seen in current session
  session_correct     Correct answers in current session

Due to compute limits on Databricks Community Edition, we used a
**stratified sample of \~1.4 million rows** while preserving user
diversity.

------------------------------------------------------------------------

## Data Cleaning and Preprocessing

The dataset required cleaning before modeling.

### Data Cleaning Steps

-   Removed duplicate rows
-   Filtered invalid records such as:
    -   `delta ≤ 0`
    -   `history_correct > history_seen`
    -   `session_correct > session_seen`

After cleaning, the dataset contained **over 12.8 million valid
records**.

------------------------------------------------------------------------

## Feature Engineering

Two behavioral features were created:

    history_accuracy = history_correct / history_seen
    session_accuracy = session_correct / session_seen

These features represent normalized measures of long‑term and short‑term
learning performance.

Redundant count-based variables were later removed to reduce
multicollinearity.

------------------------------------------------------------------------

## Exploratory Data Analysis

Several behavioral patterns were identified during EDA.

### Memory Decay

A clear relationship was observed between **time since last review
(delta)** and recall probability. Longer delays correspond to lower
recall probability, reflecting the **forgetting curve**.

### Session Performance

Users who answered more questions correctly in a session showed
significantly higher recall probabilities.

### Historical Accuracy

Long‑term performance (`history_accuracy`) also positively correlated
with recall probability.

------------------------------------------------------------------------

## Machine Learning Pipeline

The project used **Spark MLlib** to build scalable regression models.

### Feature Processing

Numerical features were normalized using:

    StandardScaler

Categorical variables were encoded using:

    StringIndexer
    OneHotEncoder

All features were combined using:

    VectorAssembler

------------------------------------------------------------------------

## Models Used

### Linear Regression

Used as a baseline model due to simplicity and interpretability.

### Random Forest Regressor

Chosen for its ability to capture **non‑linear relationships in user
learning behavior**.

------------------------------------------------------------------------

## Model Performance

  Model               RMSE    MAE     R²
  ------------------- ------- ------- ----------
  Linear Regression   0.254   0.176   0.10
  Random Forest       0.113   0.064   **0.82**

Random Forest significantly outperformed linear regression, indicating
that recall probability depends on complex behavioral patterns.

------------------------------------------------------------------------

## Feature Importance

The most influential predictors were:

1.  **session_correct**
2.  **history_accuracy**
3.  **delta (time since last review)**

This suggests that **recent performance and historical learning accuracy
are strong indicators of memory recall**.

------------------------------------------------------------------------

## Challenges

### Big Data Processing

Working with 13M+ rows on Databricks Community Edition required sampling
and caching strategies.

### Computational Limits

Grid search cross‑validation could not fully complete due to resource
limitations.

Despite this, manual tuning still produced strong model performance.

------------------------------------------------------------------------

## Future Improvements

Possible extensions include:

-   sequence models such as **LSTMs**
-   modeling **word difficulty**
-   larger hyperparameter tuning experiments
-   time-aware learning models

------------------------------------------------------------------------

## Technologies Used

-   Apache Spark
-   Spark MLlib
-   Databricks Community Edition
-   Python
-   Pandas
-   Matplotlib
-   Scikit‑learn

------------------------------------------------------------------------

## Repository Structure

    duolingo-recall-prediction-sparkml
    │
    ├── data
    ├── notebooks
    ├── reports
    ├── results
    ├── README.md
    ├── requirements.txt
    └── .gitignore

------------------------------------------------------------------------

## How to Run the Project

Clone the repository:

    git clone https://github.com/YOUR_USERNAME/duolingo-recall-prediction-sparkml

Install dependencies:

    pip install -r requirements.txt

Open the notebook:

    notebooks/duolingo_spark_ml.ipynb

Run the Spark pipeline.

------------------------------------------------------------------------

## What This Project Demonstrates

This project demonstrates:

-   handling **large‑scale datasets (13M+ rows)**
-   distributed computing with **Apache Spark**
-   building **Spark ML pipelines**
-   feature engineering for behavioral prediction
-   evaluating machine learning models on big data
