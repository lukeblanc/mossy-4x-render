#!/usr/bin/env bash
export $(grep -v '^#' .env | xargs)   # load .env
uvicorn app:app --reload --port 8000
