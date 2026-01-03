python3.11 -m venv mlzoomcamp

source mlzoomcamp-venv/bin/activate

python -m pip install --upgrade pip

python -m pip install ipykernel

python -m ipykernel install --user --name=mlzoomcamp-venv --display-name "mlzoomcamp-venv (mlzoomcamp-venv)"

python -m pip install -r requirements.txt


gunicorn --bind 0.0.0.0:8686 churn_api:app


docker build -t churn_api:1.0 .
docker run -p 8686:8686 churn-api:1.0