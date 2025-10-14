.PHONY: setup train dev up down test

setup:
	python -m pip install -r backend/requirements.txt
	(cd frontend && npm install)

train:
	python scripts/train.py

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir backend

up:
	docker compose up --build

down:
	docker compose down

test:
	pytest -q backend/tests
