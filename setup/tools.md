# Tool services

The tool environment is shared by inference and RL. Search and scraping
clients use public provider settings. Enhanced Reader can be deployed
independently of training and model-serving packages.
All commands below run from the repository root.

| Component | Role | Python dependencies |
| --- | --- | --- |
| Jina Reader | Fetch/render pages and produce raw Markdown | Separate browser/Node service, available as a Docker image |
| Raw Markdown cache | Cache Reader responses | FastAPI, HTTPX, Uvicorn |
| Enhanced Reader | Clean/cache Markdown; optionally call ReaderLM | FastAPI, HTTPX, Uvicorn |

For the Python Reader services, activate a separate service environment and
install:

```bash
python -m pip install -r setup/requirements-tools.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

None of these Python services require verl, Ray, DeepSpeed, or PyTorch.
Existing public Reader endpoints remain available without self-hosting.
For Serper, Firecrawl, OSS, and model credentials, see
[tool configuration](../README.md#1-tool-setup) in the root README.

## Enhanced Reader

The included service supports `markdown_clean`: fetch raw Markdown, clean
it with local rules, and cache the result. This mode does not call ReaderLM
and does not need a GPU. Cache hits can avoid repeated page fetches.
It keeps the same `GET /https://target-url` request shape as Jina Reader.

The single-instance example uses three services:

| Host port | Service | Implementation |
| --- | --- | --- |
| 8002 | Base Jina Reader, HTTP/1.1 | Upstream Reader image or the included Reader Dockerfile |
| 8003 | Raw Markdown cache | `utils.enhanced_reader:app`, mode `raw_only` |
| 8004 | Enhanced Reader | `utils.enhanced_reader:app`, strategy `markdown_clean` |

### Base Reader

The [official Jina Reader guide](https://github.com/jina-ai/reader#self-host-with-docker)
provides a prebuilt open-source image. Its HTTP/1.1 port is `8081`; port
`8080` serves h2c. Start it with a local HTTP/1.1 mapping:

```bash
docker run --rm --name opensearch-jina-reader \
  -p 127.0.0.1:8002:8081 ghcr.io/jina-ai/reader:oss
```

Alternatively, build the included browser/Node environment and explicitly
start Reader from its source directory:

```bash
docker build -f setup/reader.DOCKERFILE -t opensearch-vl:reader .
docker run --rm --name opensearch-jina-reader \
  -p 127.0.0.1:8002:8081 -w /workspace/reader \
  opensearch-vl:reader npm run start
```

These are alternative ways to start the same base-service port. The included
Reader Dockerfile defaults to a shell, so its `npm run start` command is
specified explicitly above. Pin the chosen upstream image/source revision
when recording a deployment for reproducibility.

### Raw Markdown cache

In another terminal, activate the Python service environment and run:

```bash
RAW_READER_URL="http://127.0.0.1:8002" \
ENHANCED_READER_MODE="raw_only" \
uvicorn utils.enhanced_reader:app --host 127.0.0.1 --port 8003
```

### Enhanced Reader service

In a third terminal, use the same Python service environment:

```bash
RAW_READER_URL="http://127.0.0.1:8002" \
RAW_MARKDOWN_READER_URL="http://127.0.0.1:8003" \
ENHANCED_READER_MODE="full" \
ENHANCED_READER_FETCH_STRATEGY="markdown_clean" \
uvicorn utils.enhanced_reader:app --host 127.0.0.1 --port 8004
```

Check local readiness, then exercise a page read:

```bash
curl http://127.0.0.1:8003/__enhanced_reader/health
curl http://127.0.0.1:8004/__enhanced_reader/health
curl -H 'Accept: application/json' http://127.0.0.1:8004/https://example.com
```

Health responses describe the local process; the page-read check verifies
that the base Reader and intermediate cache are also reachable. Cache data
lives under `.cache/enhanced_reader/` by default. `ENHANCED_READER_CACHE_DIR`
and `RAW_MARKDOWN_CACHE_DIR` can move it to a persistent data location.

### Connect inference and RL

Export these in the worker environment before starting the process:

```bash
export ENHANCED_READER_URL="http://127.0.0.1:8004/"
```

For workers on other machines, bind the Python services to a reachable
interface and use the Reader host's address. These settings select the
Reader service; they do not change search, scraping, upload, or model endpoints.

### Optional ReaderLM processing

The `markdown_first` and `parallel` strategies use an independently deployed
OpenAI-compatible ReaderLM endpoint. Configure `READERLM_API_BASE` (or
`READERLM_API_BASES`), `READERLM_MODEL_NAME`, and `READERLM_API_KEY` on the
Enhanced Reader service, then select the intended strategy. The existing
`utils/start_enhanced_reader_stack.sh` can manage a local multi-worker stack
when Node/Chromium and a built Jina Reader checkout are already installed.
It does not start a ReaderLM model server.

Self-hosting and caching can reduce repeated fetching and ReaderLM conversion
calls, while adding your own CPU/browser/network/storage workload. Tool
summarization, answer judging, search, scraping, and uploads remain separate
costs. Actual savings depend on workload, cache reuse, available resources,
and model-service usage.

The protocol and cleanup paths have offline tests. Real browser fetching,
proxy egress, Docker builds, and deployment capacity still need validation
in your environment.
