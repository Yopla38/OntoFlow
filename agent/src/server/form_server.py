"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/server/form_server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uvicorn
from typing import Optional, Dict, Any
import threading


class FormServer:
    _instance: Optional['FormServer'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FormServer, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.app = FastAPI()
            self.form_response = None
            self.form_event = asyncio.Event()
            self.setup_cors()
            self.setup_routes()
            self.server = None
            self.initialized = True

    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Pour le développement, à restreindre en production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        @self.app.post("/submit")
        async def submit_form(request: Request):
            try:
                form_data = await request.json()
                self.form_response = form_data
                self.form_event.set()
                return {"status": "success"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}

    async def start(self):
        """Démarre le serveur dans une tâche asyncio"""
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=8000,
            log_level="error"
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()

    async def stop(self):
        """Arrête le serveur"""
        if self.server:
            self.server.should_exit = True
            await self.server.shutdown()

    async def wait_for_submission(self) -> Dict[str, Any]:
        """Attend une soumission de formulaire"""
        await self.form_event.wait()
        response = self.form_response
        self.form_response = None
        self.form_event.clear()
        return response


