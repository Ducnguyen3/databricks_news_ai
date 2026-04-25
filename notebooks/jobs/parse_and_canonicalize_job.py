# Databricks notebook source
from __future__ import annotations

import sys
from pathlib import Path


def _add_project_root_to_python_path() -> None:
    try:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # type: ignore[name-defined]
        project_root = "/".join(notebook_path.split("/")[:-3])
        workspace_project_root = f"/Workspace{project_root}"
        if workspace_project_root not in sys.path:
            sys.path.insert(0, workspace_project_root)
    except Exception:
        local_project_root = Path(__file__).resolve().parents[2]
        if str(local_project_root) not in sys.path:
            sys.path.insert(0, str(local_project_root))


_add_project_root_to_python_path()

from app.jobs.parse_and_canonicalize_job import main

main()
