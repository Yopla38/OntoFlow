"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/providers/memory_providers.py
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import aiosqlite

from agent.src.types.interfaces import MemoryProvider
from typing import Dict, Any, List
import motor.motor_asyncio
import pinecone


class MongoDBMemory(MemoryProvider):
    def __init__(self, connection_string: str, database: str, collection: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db[collection]

    async def add(self, data: Dict[str, Any]):
        await self.collection.insert_one(data)

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        cursor = self.collection.find(
            {"$text": {"$search": query}}
        ).limit(k)
        return await cursor.to_list(length=k)

    async def clear(self):
        await self.collection.delete_many({})


class PineconeMemory(MemoryProvider):
    def __init__(self, api_key: str, environment: str, index_name: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    async def add(self, data: Dict[str, Any]):
        # Implement vector embedding and storage
        pass

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Implement vector search
        pass

    async def clear(self):
        # Implement clear functionality
        pass


class LocalFileMemory(MemoryProvider):
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    async def add(self, data: Dict[str, Any]):
        logging.info(f"Adding data to local file: {self.file_path}")
        #logging.info(f"Data to add: {data}")

        try:
            with open(self.file_path, 'r') as f:
                memories = json.load(f)
        except json.JSONDecodeError:
            logging.warning("Could not decode existing memories, starting fresh")
            memories = []

        logging.info(f"Current memories count: {len(memories)}")
        memories.append(data)

        with open(self.file_path, 'w') as f:
            json.dump(memories, f, indent=2, default=str)

        logging.info(f"New memories count: {len(memories)}")

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r') as f:
            memories = json.load(f)

        # Recherche simple basée sur la correspondance de chaînes
        # Vous pourriez améliorer cela avec une recherche plus sophistiquée
        matched_memories = [
            m for m in memories
            if query.lower() in str(m).lower()
        ]

        return matched_memories[:k]

    async def clear(self):
        with open(self.file_path, 'w') as f:
            json.dump([], f)


class SQLiteMemory(MemoryProvider):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def add(self, data: Dict[str, Any]):
        async with aiosqlite.connect(self.db_path) as db:
            content = json.dumps(data.get('content', ''))
            metadata = json.dumps(data.get('metadata', {}))
            await db.execute(
                "INSERT INTO memories (content, metadata) VALUES (?, ?)",
                (content, metadata)
            )
            await db.commit()

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT content, metadata, timestamp 
                FROM memories 
                WHERE content LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (f"%{query}%", k)
            )
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    'content': json.loads(row[0]),
                    'metadata': json.loads(row[1]),
                    'timestamp': row[2]
                })

            return results

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM memories")
            await db.commit()


class InMemoryStorage(MemoryProvider):
    def __init__(self, max_size: int = 1000):
        self.memories: List[Dict[str, Any]] = []
        self.max_size = max_size

    async def add(self, data: Dict[str, Any]):
        data['timestamp'] = datetime.now().isoformat()
        self.memories.append(data)

        # Maintenir la taille maximale
        if len(self.memories) > self.max_size:
            self.memories.pop(0)

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Recherche simple avec correspondance de chaînes
        matched_memories = [
            m for m in self.memories
            if query.lower() in str(m).lower()
        ]

        # Trier par timestamp décroissant
        matched_memories.sort(
            key=lambda x: datetime.fromisoformat(x['timestamp']),
            reverse=True
        )

        return matched_memories[:k]

    async def clear(self):
        self.memories.clear()


# indexation vectorielle locale
class LocalVectorMemory(MemoryProvider):
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.memories: List[Dict[str, Any]] = []

    async def add(self, data: Dict[str, Any]):
        content = str(data.get('content', ''))
        embedding = self.model.encode([content])[0]

        self.index.add(embedding.reshape(1, -1))
        data['timestamp'] = datetime.now().isoformat()
        self.memories.append(data)

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query])[0]

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.memories):
                results.append(self.memories[idx])

        return results

    async def clear(self):
        self.index.reset()
        self.memories.clear()



