.PHONY: help test up down deploy-modal

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

test: ## Run unit tests
	PYTHONPATH=. python3 -m unittest discover tests/

up: ## Start local Redis and services via Docker Compose
	docker compose up --build -d

down: ## Stop Docker Compose services
	docker compose down

deploy-modal: ## Deploy GPU functions to Modal
	modal deploy inference/modal_app.py
