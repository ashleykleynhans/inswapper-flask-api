# Changelog

## [2.0.1] - 2026-07-16

### Changed
- Renamed project from inswapper-flask-api to FaceSwap API (faceswap-api)
- Updated GitHub repository description and tags

### Fixed
- Fix Dockerfile: clone CodeFormer before running download_models.py to avoid directory conflict
- Fix CI: add missing pytest-asyncio to requirements.txt
- Fix CI: add asyncio_mode = auto to pytest.ini
- Fix CI: mock cv2 in restoration tests to avoid real OpenCV calls

## [2.0.0] - 2026-07-15

### Added
- Migrated from Flask to FastAPI with uvicorn, async lifespan, and auto-generated OpenAPI docs at `/docs`
- VRAM-safe async job queue (asyncio.Queue with single serial worker) — processes one request at a time to prevent GPU OOM
- 13 face swap models: inswapper, simswap, ghost, hififace, hyperswap, blendswap, uniface with per-model metadata
- docker-bake.hcl with provenance and SBOM attestations
- GitHub Actions CI workflow: test, detect changes, build and push to ghcr.io
- Dependabot configuration for pip, Docker, and GitHub Actions
- Enhanced FaceFusion-based swap pipeline: cv2.estimateAffinePartial2D + RANSAC warp, soft-edge box mask paste-back, 4 source preparation strategies, identity blending
- Face selector: 7 sort orders (left-right, right-left, top-bottom, small-large, large-small, best-worst, worst-best), gender filter (male/female), age range filter
- Configurable mask controls: per-edge padding and blur
- Configurable face swap resolution per model
- Lazy model loading with caching — only loads model on first use
- Model download script: downloads all 13 ONNX files, 3 embedding converters, insightface buffalo_l, and 4 CodeFormer weight files
- Comprehensive test suite: 203 tests, zero warnings
- `pytest.ini` and `.coveragerc` with coverage configured by default

### Changed
- License changed from GPL v3 to OpenRAIL-AS (matching FaceFusion)
- `GET /` health endpoint now returns available models and queue depth
- `POST /faceswap` returns `202 Accepted` with a job ID (async queue mode)
- `POST /faceswap/sync` blocks until complete (backward compatible mode)
- Updated Dockerfile: CUDA 12.4.1 base image, PyTorch with CUDA 12.4 wheels, onnxruntime-gpu, models pre-downloaded in build
- Enhanced CodeFormer restoration with graceful GPU error degradation and automatic upscale limiting for large images
- Reorganized into `app/` package with services/routes/models/queue separation

### Removed
- Flask and waitress dependencies
- Old `app.py` and `restoration.py` flat files (replaced by package)

### Fixed
- CodeFormer restoration error handling: gracefully falls back to original crop on GPU errors
