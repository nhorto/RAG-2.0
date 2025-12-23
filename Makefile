.PHONY: install setup ingest run-ui clean help

help:
	@echo "Tekla PowerFab RAG System v2.0"
	@echo ""
	@echo "Available commands:"
	@echo "  make install              Install Python dependencies"
	@echo "  make setup                Set up Qdrant and environment"
	@echo "  make ingest               Ingest documents into vector database"
	@echo "  make run-ui               Run Streamlit UI"
	@echo "  make clean                Clean generated files"
	@echo "  make test                 Run tests"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	python -c "import spacy; spacy.cli.download('en_core_web_sm')"
	@echo "Done! Don't forget to configure your .env file"

setup:
	@echo "Setting up Qdrant (Docker)..."
	docker run -d -p 6333:6333 -p 6334:6334 \
		-v $$(pwd)/qdrant_storage:/qdrant/storage:z \
		--name qdrant \
		qdrant/qdrant
	@echo "Creating .env file..."
	cp .env.example .env
	@echo "Done! Edit .env with your API keys"

ingest:
	@echo "Ingesting documents..."
	python scripts/ingest_documents.py \
		--source-dir data/raw \
		--collection consulting_transcripts

run-ui:
	@echo "Starting Streamlit UI..."
	streamlit run ui/streamlit_app.py

test:
	@echo "Running tests..."
	pytest tests/

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	rm -rf data/embeddings_cache/*
	rm -rf logs/*
	@echo "Done!"

stop-qdrant:
	@echo "Stopping Qdrant..."
	docker stop qdrant
	docker rm qdrant

restart-qdrant:
	@echo "Restarting Qdrant..."
	docker restart qdrant
