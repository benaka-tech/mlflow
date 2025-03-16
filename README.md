# Credit Card Fraud Detection

This project addresses the high-impact issue of detecting fraudulent credit card transactions, which is critical for minimizing financial losses and protecting consumers. The project involves designing and developing a data pipeline and a machine learning pipeline, and providing API access to key application details.

## Sub-Objectives

### Sub-Objective 1: Design and Development of a Data Pipeline
**Weight: 8 marks**

#### Activities
1. **Business Understanding**: Identify a business problem in the area of data science.
2. **Data Ingestion**: Find an appropriate dataset from a public repository (e.g., Kaggle). Ensure that the dataset has sufficient records to carry out a meaningful data science experiment.
3. **Data Pre-processing**: Perform activities such as displaying summary statistics, checking for missing values, imputing missing data for numeric columns, displaying data types, and normalizing data.
4. **Exploratory Data Analysis (EDA)**: Conduct EDA to include calculating correlation coefficients, identifying correlations between numeric and/or categorical features, binning, encoding, assessing feature importance, and visualizing data (using charts and graphs for univariate and bivariate analyses).
5. **DataOps**: Implement workflows to automate activities from steps 1.3 and 1.4 within a data pipeline. Schedule these workflows to run every 3 minutes, logging all activity details and displaying them on a Cloud dashboard.

### Sub-Objective 2: Design and Development of a Machine Learning Pipeline
**Weight: 5 marks**

#### Activities
1. **Model Preparation**: Identify suitable machine learning algorithms for solving the business problem based on the dataset. Select any two algorithms.
2. **Model Training**: Split the dataset into training (70%) and testing (30%) sets and train the models.
3. **Model Evaluation**: Evaluate the models using at least one metric (e.g., accuracy for classification models).
4. **MLOps**: Monitor the model and log relevant metrics (at least four, such as accuracy, precision, recall, F1 score, etc.).

### Sub-Objective 3: API Access
**Weight: 2 marks**

#### Activities
1. **Retrieve Key Application Details**: Use built-in APIs to access important application information (e.g., flow, deployment, etc.).
2. **Display Application Details**: Present at least four application details retrieved via APIs.

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
Here is the updated README.md file with the setup instructions added:

```markdown
# Credit Card Fraud Detection

This project addresses the high-impact issue of detecting fraudulent credit card transactions, which is critical for minimizing financial losses and protecting consumers. The project involves designing and developing a data pipeline and a machine learning pipeline, and providing API access to key application details.

## Sub-Objectives

### Sub-Objective 1: Design and Development of a Data Pipeline
**Weight: 8 marks**

#### Activities
1. **Business Understanding**: Identify a business problem in the area of data science.
2. **Data Ingestion**: Find an appropriate dataset from a public repository (e.g., Kaggle). Ensure that the dataset has sufficient records to carry out a meaningful data science experiment.
3. **Data Pre-processing**: Perform activities such as displaying summary statistics, checking for missing values, imputing missing data for numeric columns, displaying data types, and normalizing data.
4. **Exploratory Data Analysis (EDA)**: Conduct EDA to include calculating correlation coefficients, identifying correlations between numeric and/or categorical features, binning, encoding, assessing feature importance, and visualizing data (using charts and graphs for univariate and bivariate analyses).
5. **DataOps**: Implement workflows to automate activities from steps 1.3 and 1.4 within a data pipeline. Schedule these workflows to run every 3 minutes, logging all activity details and displaying them on a Cloud dashboard.

### Sub-Objective 2: Design and Development of a Machine Learning Pipeline
**Weight: 5 marks**

#### Activities
1. **Model Preparation**: Identify suitable machine learning algorithms for solving the business problem based on the dataset. Select any two algorithms.
2. **Model Training**: Split the dataset into training (70%) and testing (30%) sets and train the models.
3. **Model Evaluation**: Evaluate the models using at least one metric (e.g., accuracy for classification models).
4. **MLOps**: Monitor the model and log relevant metrics (at least four, such as accuracy, precision, recall, F1 score, etc.).

### Sub-Objective 3: API Access
**Weight: 2 marks**

#### Activities
1. **Retrieve Key Application Details**: Use built-in APIs to access important application information (e.g., flow, deployment, etc.).
2. **Display Application Details**: Present at least four application details retrieved via APIs.

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install "flask[async]" prefect mlflow scikit-learn pandas matplotlib seaborn
   ```

## Running the Project

1. **Start the Prefect Server**:
   ```sh
   prefect server start --host 0.0.0.0 --cors-allowed-origins "*"
   ```

2. **Set the Prefect API URL**:
   ```sh
   prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   ```

3. **Start the MLflow Server**:
   ```sh
   mlflow ui --host 0.0.0.0 --port 5001
   ```

4. **Run the Flask Application**:
   ```sh
   python app.py
   ```

## Accessing the UIs

- **Prefect Server UI**: Open your web browser and navigate to `http://127.0.0.1:4200`.
- **MLflow UI**: Open your web browser and navigate to `http://127.0.0.1:5001`.

## API Endpoints

- **Pipeline Details**: `GET /pipeline_details`
  - Retrieves key pipeline details such as MLflow run IDs, model performance metrics, feature importance plot file, and timestamp of the pipeline run.

- **Flow Details**: `GET /flow_details`
  - Retrieves flow details from Prefect.

- **Deployment Details**: `GET /deployment_details`
  - Retrieves deployment details from Prefect.

- **Task Run Details**: `GET /task_run_details`
  - Retrieves task run details from Prefect.

## Project Structure

```
.
├── app.py                  # Main application file
├── README.md               # Project README file
├── requirements.txt        # List of required packages
└── data                    # Directory to store the dataset
    └── creditcard_2023.csv # Credit Card Fraud Detection dataset
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### Explanation of the README File:
1. **Project Overview**: Provides a brief overview of the project and its objectives.
2. **Sub-Objectives**: Details the activities for each sub-objective.
3. **Installation**: Instructions for setting up the project.
4. **Running the Project**: Steps to start the Prefect server, set the Prefect API URL, start the MLflow server, and run the Flask application.
5. **Accessing the UIs**: URLs to access the Prefect Server UI and MLflow UI.
6. **API Endpoints**: Describes the available API endpoints and their purposes.
7. **Project Structure**: Provides an overview of the project directory structure.
8. **License**: Information about the project license.

