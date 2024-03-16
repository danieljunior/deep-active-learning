CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document (
  id SERIAL PRIMARY KEY,
  embedding vector,
  text text,
  created_at timestamptz DEFAULT now()
);