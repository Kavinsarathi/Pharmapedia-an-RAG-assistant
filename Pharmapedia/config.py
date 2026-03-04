from dotenv import load_dotenv
import os
load_dotenv()

DATA_PATH = "D:\Pharmapedia_An_AI_Assistant\doc\drug-label-0001-of-0013.json"
drug_file_path = "D:\Pharmapedia_An_AI_Assistant\Pharmapedia\drugs.txt"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
Persistant_dir='D:\Pharmapedia_An_AI_Assistant\Pharmapedia\chromadb'
collection_name = 'pharmapedia_embed_by_BAAI'
pg_connection=f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_Name')}"