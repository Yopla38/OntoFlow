"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# communicative_server.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import asyncio
from threading import Thread
import logging
from queue import Queue


app = Flask(__name__)
# Changement du mode async
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")


class CommunicationServer:
    def __init__(self):
        self.agents = {}
        self.server_thread = None
        self.is_running = False
        self.socketio = socketio

    def register_agent(self, agent_id: str, agent):
        """Enregistre un agent pour la communication"""
        # Configure l'agent avec l'instance socketio
        if hasattr(agent, 'socketio'):
            agent.socketio = self.socketio
        self.agents[agent_id] = agent

    def setup_routes(self):
        @app.route('/')
        def home():
            return render_template('chat.html')

        @socketio.on('connect')
        def handle_connect():
            print(f"Client connected to communication server")

        @socketio.on('message')
        def handle_message(data):
            try:
                agent_id = data.get('agent_id')
                message = data.get('message')

                if agent_id not in self.agents:
                    socketio.emit('error', "Agent non trouvé")
                    return

                agent = self.agents[agent_id]

                # Créer une nouvelle boucle d'événements pour cette requête
                async def process_message():
                    try:
                        await agent.communicate(message)
                    except Exception as e:
                        socketio.emit('error', str(e))

                # Exécuter la coroutine dans une nouvelle boucle
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(process_message())
                finally:
                    loop.close()

            except Exception as e:
                logging.error(f"Erreur dans handle_message: {e}")
                socketio.emit('error', str(e))

        @socketio.on('status')
        def handle_status(data):
            agent_id = data.get('agent_id')
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                return agent.get_status_report()
            return {"error": "Agent non trouvé"}

    def start(self, host='0.0.0.0', port=5000):
        """Démarre le serveur dans un thread séparé"""
        self.setup_routes()
        self.is_running = True

        def run_server():
            socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def stop(self):
        """Arrête le serveur"""
        self.is_running = False
        if self.server_thread:
            self.server_thread.join(timeout=1)


# Instance globale du serveur
communication_server = CommunicationServer()