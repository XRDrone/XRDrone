
# Contributing

Thank you for contributing to XRDrone. This document describes the workflow for development,
testing, documentation updates, and submitting pull requests.

The repository uses a docs-first structure, so detailed documentation lives in the `docs/`
directory rather than in the top-level README.

---

# 1. Development Environment

Before contributing, set up the Python environment using the setup guide.

Recommended steps:

python3 -m venv yolovenv
source yolovenv/bin/activate
pip install -r requirements.txt

If CUDA support is required:

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

See: docs/setup.md

---

# 2. Repository Documentation

Main documentation files:

docs/setup.md – environment setup  
docs/settings.md – runtime configuration settings  
docs/udp-json.md – UDP packet schema used for Unity communication  
docs/runtime-ui-and-terminal-reference.md – runtime overlay and terminal text reference  
docs/testing.md – UDP validation and testing procedures

Contributors should review these files before making structural changes.

---

# 3. Code Style

The repository uses Ruff for linting and formatting.

Run before committing:

ruff check .
ruff format .

Optional automatic hook:

pre-commit install

---

# 4. Branch Workflow

Create a feature branch:

git checkout -b feature/your-feature

Make your changes, run checks, then commit:

git add .
git commit -m "Short descriptive message"

Push the branch:

git push origin feature/your-feature

Then open a Pull Request.

---

# 5. Testing Changes

Testing instructions are documented in docs/testing.md.

Typical validation workflow:

1. Run the pipeline (python main.py)
2. Verify detections and pose behavior
3. Confirm UDP packets match schema
4. Confirm Unity receives detections and pose data

Algorithm changes should include a reproducible demonstration.

---

# 6. UDP JSON Protocol Changes

Changes to the UDP schema must:

1. Be documented in docs/udp-json.md
2. Maintain compatibility with Unity when possible
3. Be validated using test_with_coverage.py

Unity components depend on this schema remaining stable.

---

# 7. Configuration Changes

Runtime behavior is controlled by settings.py.

Examples:

- detection thresholds
- smoothing parameters
- pose configuration
- adaptive tuning parameters

If new configuration options are added:

1. Add them to settings.py
2. Document them in docs/settings.md
3. Provide default values

---

# 8. Documentation Updates

Documentation must stay synchronized with implementation.

Update documentation when:

- pipeline behavior changes
- settings are added or removed
- runtime overlays or terminal outputs change
- UDP packet structure changes

Relevant documentation:

docs/setup.md  
docs/settings.md  
docs/udp-json.md  
docs/runtime-ui-and-terminal-reference.md  
docs/testing.md

---

# 9. Pull Request Requirements

Before submitting a PR ensure:

- Code passes ruff check
- Code is formatted (ruff format)
- Pipeline runs without runtime errors
- Documentation is updated if behavior changes
- No unnecessary files are committed

Do not commit:

.DS_Store  
__pycache__/  
temporary media files  
local environment files

---

# 10. Commit Message Guidelines

Examples:

Add motion smoothing for ArUco pose updates  
Optimize UDP packet serialization  
Implement adaptive runtime tuning  
Update documentation for runtime overlays

---

# 11. Questions

If unsure about architecture decisions or major pipeline changes,
open an issue before implementing large modifications.
