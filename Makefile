OPENAI_ENV := .env
SRC_DIR := src
LANGFUSE_DIR := $(SRC_DIR)/core/observability
LANGFUSE_ENV := $(LANGFUSE_DIR)/.env.langfuse
COMPOSE := docker compose --env-file $(LANGFUSE_ENV) -f $(LANGFUSE_DIR)/docker-compose.langfuse.yml
OPENWEBUI_DIR := $(SRC_DIR)/core/openwebui
OPENWEBUI_COMPOSE := docker compose --env-file $(OPENAI_ENV) -f $(OPENWEBUI_DIR)/docker-compose.openwebui.yml

.PHONY: init test demo-all langfuse-up langfuse-down langfuse-logs langfuse-demo demo-file demo-context demo-core demo-run-command demo-policy-gate demo-langfuse openwebui-up openwebui-down openwebui-logs

# Install/sync Python dependencies from uv.lock.
init: $(OPENAI_ENV) $(LANGFUSE_ENV)
	uv sync

# Create project-root OpenAI env file from template on first run.
$(OPENAI_ENV):
	cp .env.example $(OPENAI_ENV)

# Run the full test suite.
test: init
	uv run pytest -q

# Run OpenAI-backed no-docker demos in sequence.
demo-all: init $(LANGFUSE_ENV)
	uv run python $(SRC_DIR)/demos/demo.py
	uv run python $(SRC_DIR)/demos/demo_context.py
	uv run python $(SRC_DIR)/demos/demo_core.py
	uv run python $(SRC_DIR)/demos/demo_run_command.py
	uv run python $(SRC_DIR)/demos/demo_policy_gate.py

# Create local env file from template on first run.
$(LANGFUSE_ENV):
	cp $(LANGFUSE_DIR)/.env.langfuse.example $(LANGFUSE_ENV)

# Start local Langfuse (http://localhost:3000).
langfuse-up: $(LANGFUSE_ENV)
	$(COMPOSE) up -d

# Stop local Langfuse.
langfuse-down:
	$(COMPOSE) down

# Tail Langfuse container logs.
langfuse-logs:
	$(COMPOSE) logs -f

# Start OpenWebUI with the agent Pipe (http://localhost:3001). Setup: src/core/openwebui/README.md.
# Host port 3001 by default (coexists with Langfuse on 3000); override: OPENWEBUI_PORT=<port> make openwebui-up
openwebui-up: $(OPENAI_ENV)
	$(OPENWEBUI_COMPOSE) up -d

# Stop OpenWebUI.
openwebui-down:
	$(OPENWEBUI_COMPOSE) down

# Tail OpenWebUI container logs.
openwebui-logs:
	$(OPENWEBUI_COMPOSE) logs -f

# Publish a demo trace (token/cost) to local Langfuse.
langfuse-demo: init langfuse-up
	uv run python $(SRC_DIR)/demos/demo_langfuse.py

# Backward-compatible alias for the Langfuse canned demo.
demo-langfuse: langfuse-demo

# Run the OpenAI-backed file/tool demo.
demo-file: init $(LANGFUSE_ENV)
	uv run python $(SRC_DIR)/demos/demo.py

# Run the OpenAI-backed context-budgeting demo.
demo-context: init $(LANGFUSE_ENV)
	uv run python $(SRC_DIR)/demos/demo_context.py

# Run the OpenAI-backed core telemetry/summarization demo.
demo-core: init $(LANGFUSE_ENV)
	uv run python $(SRC_DIR)/demos/demo_core.py

# Run the OpenAI-backed run_command demo.
demo-run-command: init $(LANGFUSE_ENV)
	uv run python $(SRC_DIR)/demos/demo_run_command.py

# Run the OpenAI-backed HITL policy-gate demo.
demo-policy-gate: init $(LANGFUSE_ENV)
	uv run python $(SRC_DIR)/demos/demo_policy_gate.py
