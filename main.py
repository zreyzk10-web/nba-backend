import os
import json
import requests
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Cargar variables de entorno si existen localmente (.env)
load_dotenv()

# ==========================================
# 1. CONFIGURACIÓN Y CLIENTES
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BDL_API_KEY = os.getenv("BDL_API_KEY", "")  # Opcional si usas BallDontLie

# Inicializar cliente oficial de Gemini
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

app = FastAPI(
    title="NBA AI Predictor API",
    description="API Backend para pronósticos estadísticos de la NBA generados por IA",
    version="1.0.0"
)

# Habilitar CORS para que la App Móvil (Flutter/React) pueda consumir la API sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. MODELOS PYDANTIC (Esquema de Datos)
# ==========================================
class PlayerProp(BaseModel):
    player_name: str = Field(description="Nombre completo del jugador")
    market: str = Field(description="Mercado recomendado (ej. OVER 24.5 Puntos)")
    justification: str = Field(description="Explicación estadística clave en 1 frase")

class PredictionRequest(BaseModel):
    home_team: str = Field(example="Boston Celtics")
    visitor_team: str = Field(example="New York Knicks")
    date: Optional[str] = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

class PredictionResponse(BaseModel):
    matchup: str = Field(description="Ejemplo: New York Knicks @ Boston Celtics")
    predicted_winner: str = Field(description="Equipo ganador proyectado")
    predicted_margin: str = Field(description="Margen proyectado de victoria (ej. 4 a 7 puntos)")
    estimated_total_points: float = Field(description="Total acumulado estimado de puntos")
    over_under_recommendation: str = Field(description="OVER o UNDER para la línea de 220.5")
    confidence_score: float = Field(description="Puntaje de confianza del 1.0 al 10.0")
    key_reasons: List[str] = Field(description="Tres argumentos estadísticos principales")
    player_props: List[PlayerProp] = Field(description="Dos mejores props de jugadores")
    risk_factors: List[str] = Field(description="Puntos de riesgo a monitorear antes del partido")


# ==========================================
# 3. ENDPOINTS DE LA API
# ==========================================

@app.get("/")
def read_root():
    """Endpoint de salud/verificación."""
    return {
        "status": "online",
        "service": "NBA AI Predictor Backend",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/games/today")
def get_today_games():
    """Obtiene la lista de partidos programados para la fecha actual."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Si tienes API Key de BallDontLie activa
    if BDL_API_KEY:
        url = f"https://api.balldontlie.io/v1/games?dates[]={today}"
        headers = {"Authorization": BDL_API_KEY}
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass  # Si falla la API externa, usamos la lista por defecto
    
    # Cartelera estática de respaldo para pruebas rápidas
    return {
        "data": [
            {"id": 101, "home_team": "Boston Celtics", "visitor_team": "New York Knicks", "time": "7:00 PM ET"},
            {"id": 102, "home_team": "Orlando Magic", "visitor_team": "Minnesota Timberwolves", "time": "7:00 PM ET"},
            {"id": 103, "home_team": "Miami Heat", "visitor_team": "Detroit Pistons", "time": "8:00 PM ET"},
            {"id": 104, "home_team": "Los Angeles Lakers", "visitor_team": "Phoenix Suns", "time": "10:00 PM ET"}
        ]
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def generate_prediction(req: PredictionRequest):
    """
    Genera un pronóstico detallado con Gemini basado en los dos equipos solicitados.
    """
    if not gemini_client:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY no está configurada en las variables de entorno."
        )

    prompt = f"""
    Actúa como un analista experto en estadísticas avanzadas y apuestas de la NBA para la temporada actual (2026).
    Genera un pronóstico profundo y objetivo para este partido:
    
    - Equipo Local: {req.home_team}
    - Equipo Visitante: {req.visitor_team}
    - Fecha: {req.date}

    Factores a evaluar:
    1. Ritmo de juego (Pace), Offensive y Defensive Ratings recientes.
    2. Descanso acumulado, historial reciente de local/visitante.
    3. Fortaleza perimetral vs triples del rival y batalla de rebotes en la pintura.

    Sigue estrictamente el esquema JSON solicitado.
    """

    try:
        # Llamada estructurada a Gemini 2.5 Flash
        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PredictionResponse,
                temperature=0.2,  # Respuesta consistente y centrada
            ),
        )

        # Parsear la respuesta en un diccionario
        prediction_dict = json.loads(response.text)
        return PredictionResponse(**prediction_dict)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el pronóstico con la IA: {str(e)}"
        )
