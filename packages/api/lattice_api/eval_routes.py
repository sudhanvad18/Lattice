"""Benchmark results produced by ``lattice eval``."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from lattice.config import get_settings
from lattice.evaluation import load_report
from lattice_api.models import EvalRunResponse


def eval_results_path() -> Path:
    """Location of the latest benchmark run; overridable in tests."""
    return Path(get_settings().data_dir) / "eval" / "latest.json"


router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/results", response_model=EvalRunResponse)
def get_eval_results(path: Annotated[Path, Depends(eval_results_path)]):
    report = load_report(path)
    if report is None:
        return EvalRunResponse()
    return EvalRunResponse.model_validate(report)
