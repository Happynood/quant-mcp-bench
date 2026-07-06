FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv==0.4.30

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN uv sync --no-dev

# U0 (the in-repo toy server) needs nothing beyond Python. Reference server
# tiers launched via `npx`/`uvx` (filesystem, git, memory) need Node.js and
# uv's tool-runner on PATH too — add them here once those tiers are wired in.

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["quantmcp"]
CMD ["--help"]
