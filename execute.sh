#!/usr/bin/bash

# This script is used to execute the program
py ./ingest_documents.py
py ./vector_search.py
py ./generate_response.py
py ./app.py

