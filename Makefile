.PHONY: install test lint preprocess train serve up down clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	flake8 src --max-line-length=110 --extend-ignore=E203

preprocess:
	cd src && python preprocess.py

train:
	cd src && python train.py

serve:
	cd src && uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

# Full local stack: MLflow + train job + serving API + Prometheus
up:
	docker compose up --build

down:
	docker compose down -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/processed/* .pytest_cache
