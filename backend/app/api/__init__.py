"""HTTP API routers."""

from fastapi import APIRouter

from app.api import (
    routes_agents,
    routes_auth,
    routes_discovery,
    routes_figures,
    routes_graph,
    routes_jobs,
    routes_lbd,
    routes_matrix,
    routes_papers,
    routes_reader,
    routes_reports,
    routes_system,
    routes_verify,
)

api_router = APIRouter()
api_router.include_router(routes_system.router)
api_router.include_router(routes_auth.router)
api_router.include_router(routes_papers.router)
api_router.include_router(routes_jobs.router)
api_router.include_router(routes_figures.router)
api_router.include_router(routes_agents.router)
api_router.include_router(routes_graph.router)
api_router.include_router(routes_reports.router)
api_router.include_router(routes_reader.router)
api_router.include_router(routes_matrix.router)
api_router.include_router(routes_discovery.router)
api_router.include_router(routes_lbd.router)
api_router.include_router(routes_verify.router)

__all__ = ["api_router"]
