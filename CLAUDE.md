# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiroFish is a multi-agent swarm intelligence prediction engine. It builds knowledge graphs from seed data, simulates thousands of AI agents interacting on virtual Twitter/Reddit platforms (via CAMEL-OASIS), and generates analytical reports — all to predict outcomes of real-world scenarios.

## Commands

### Setup
```bash
npm run setup:all        # Install all dependencies (frontend + backend)
npm run setup            # Frontend npm install only
npm run setup:backend    # Backend: uv sync (Python deps)
```

### Development
```bash
npm run dev              # Run backend + frontend concurrently
npm run backend          # Backend only: Flask on port 5001
npm run frontend         # Frontend only: Vite on port 3000
```

### Build
```bash
npm run build            # Build frontend (Vite)
```

### Backend (Python)
```bash
cd backend && uv run python run.py          # Start Flask server
cd backend && uv run python -m pytest       # Run tests (if any)
```

### Docker
```bash
docker-compose up        # Full stack via Docker
```

## Architecture

### Stack
- **Backend**: Python ≥3.11 Flask 3.0, managed by `uv`
- **Frontend**: Vue 3 + Vite, port 3000; proxies `/api` → port 5001
- **LLM**: OpenAI SDK-compatible (default: Qwen via `dashscope`; also works with GLM, OpenAI)
- **Memory/Graph**: Zep Cloud (knowledge graph for entity storage and retrieval)
- **Simulation**: CAMEL-OASIS (multi-agent Twitter + Reddit simulation)
- **Visualization**: D3.js

### Required Environment Variables
Copy `.env.example` to `.env`:
```
LLM_API_KEY         # Required
LLM_BASE_URL        # Default: https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME      # Default: qwen-plus
ZEP_API_KEY         # Required (Zep Cloud)
```

### 5-Step Pipeline
The core workflow is a sequential async pipeline:
1. **Graph Build** — Upload files → LLM extracts ontology → Zep Cloud builds knowledge graph
2. **Env Setup** — Read Zep entities → Generate OASIS agent profiles (AI personalities)
3. **Simulation** — CAMEL-OASIS runs agents on dual platforms (Twitter + Reddit) in parallel
4. **Report** — ReportAgent (ReACT loop) queries graph with tools: `SearchResult`, `InsightForge`, `Panorama`, `Interview`
5. **Interaction** — Chat with simulated agents or the ReportAgent

### Backend Structure (`backend/app/`)
- `api/` — Flask blueprints: `graph_bp`, `simulation_bp`, `report_bp`
- `services/` — Core logic: graph building, simulation runner, report agent, Zep tools
- `models/` — `Project` and `Task` state objects (in-memory, JSON-serializable)
- `utils/` — LLM client wrapper, file parser, retry logic, Zep pagination
- `config/config.py` — All configuration (LLM, Zep, chunking, simulation params)

Long-running operations (ontology generation, graph build, profile generation, report generation) run as background tasks tracked via `Task` objects with progress polling.

### Frontend Structure (`frontend/src/`)
- `views/` — Page components mapped to routes; `Process.vue` is the main 50KB workflow orchestrator
- `components/` — `Step1-5` step components + `GraphPanel.vue` (D3 graph visualization)
- `api/` — Axios services (`graph.js`, `simulation.js`, `report.js`) with 5-min timeout and exponential retry

### Key Implementation Details
- Reasoning model outputs (e.g., MiniMax/GLM with `<think>` tags or markdown code fences) are stripped before processing — see recent fix in commit `985f89f`
- Simulation state is managed in `SimulationManager`; IPC between processes via `simulation_ipc.py`
- Interview/chat with agents uses prefix injection to suppress tool calls in responses
- Default simulation: max 10 rounds, Twitter actions include CREATE_POST/LIKE/REPOST/FOLLOW/QUOTE/DO_NOTHING; Reddit adds CREATE_COMMENT/DISLIKE
