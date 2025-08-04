# Lokal RAG med LM Studio + Qdrant

## Kom igång

1. Starta Qdrant:
   docker compose up -d

2. Installera beroenden:
   pip install -r requirements.txt

3. Lägg dina dokument i data/

4. Kör pipelinen:
   python scripts/pipeline.py

5. Kör frågemotorn:
   python scripts/query.py

6. Stoppa Qdrant:
   docker compose down
   docker compose down -v (för att även ta bort volymer)

OBS
LM Studio måste vara igång med en startad modell
API-server måste vara aktiverad i LM Studio