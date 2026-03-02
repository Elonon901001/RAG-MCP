# Smart Knowledge Hub

Project scaffold initialized for DEV_SPEC section 5.2.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m compileall src
python -c "import mcp_server; import core; import ingestion; import libs; import observability"
python main.py
```
