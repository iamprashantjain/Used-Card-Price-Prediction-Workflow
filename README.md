# Used Card Price Prediction Workflow

## Stages of the MLOps Project

**Pre-Preparation**

- Clone from Common folder structure
- Run init_setup.sh file
- Activate venv
- Download Data: https://www.kaggle.com/competitions/playground-series-s3e8/data
- Perform Cleaning, EDA, Feature Engineering in jupyterfile






1. **Build: Model Development**
   - **Tools**: Git, GitHub, Python
   - **Activities**: Develop the machine learning model, use Git for version control, and push the code to GitHub.

2. **Test: Unit & Integration Testing**
   - **Tools**: Pytest, Tox
   - **Activities**: Write unit tests and integration tests to ensure code quality, using Tox for automation across different environments.

3. **Deliver: Docker Image Creation**
   - **Tools**: Docker, GitHub Actions
   - **Activities**: Create a Dockerfile, automate the image build process with GitHub Actions, and store the image in an Azure Container Registry.

4. **Deploy: CI/CD on Azure**
   - **Tools**: Azure DevOps
   - **Activities**: Set up CI/CD pipelines to automate deployment to Azure services.

5. **Monitoring: Post-Deployment Monitoring**
   - **Tools**: Evidently AI, Apache Airflow
   - **Activities**: Monitor model performance with Evidently AI and use Airflow for scheduling and managing workflows.

6. **Re-training: Continuous Improvement**
   - **Activities**: Set up retraining processes to update the model with new data and automate this in your pipeline.

## Additional Components

7. **Data Management: DVC (Data Version Control)**
   - **Purpose**: DVC helps you manage your datasets and model files by tracking changes and versions. It integrates seamlessly with Git, allowing you to version control data alongside your code.

8. **Experiment Tracking: MLflow**
   - **Purpose**: MLflow is used to track experiments, including parameters, metrics, and artifacts. It provides a central repository for all your model training experiments, enabling better reproducibility and collaboration.

9. **Model Registry: DagsHub**
   - **Purpose**: DagsHub acts as a model registry where you can store and manage different versions of your trained models. It allows you to keep track of model lineage and serves as a single source of truth for your models.

10. **BentoML**
    - **Purpose**: BentoML is a framework for packaging and serving machine learning models as APIs. It simplifies the process of deploying models to production by providing tools for building REST APIs around your models, making it easy to serve them in a production environment.

## Getting Started
To get started with this project, clone the repository and follow the setup instructions in the respective sections.