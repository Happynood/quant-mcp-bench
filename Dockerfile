FROM python:3.11-slim

WORKDIR /app

# The filesystem tier launches the reference MCP server via `npx`, so
# Node.js is a real runtime dependency here, not just a dev convenience.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.4.30

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN uv sync --no-dev

# U0 (in-repo) and U3 sqlite (self-written FastMCP wrapper) need nothing
# beyond Python. U1 filesystem needs the Node.js/npx installed above. U2
# git is launched via `uvx`, which ships with `uv` (installed above) and
# needs no separate runtime.

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["quantmcp"]
CMD ["--help"]
