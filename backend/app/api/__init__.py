"""HTTP API routers."""

from fastapi import APIRouter

from app.api import routes_agents, routes_graph, routes_papers, routes_reports, routes_system

api_router = APIRouter()
api_router.include_router(routes_system.router)
api_router.include_router(routes_papers.router)
api_router.include_router(routes_agents.router)
api_router.include_router(routes_graph.router)
api_router.include_router(routes_reports.router)

__all__ = ["api_router"]
