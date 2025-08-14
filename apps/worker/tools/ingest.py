from typing import Any, Dict
from pydantic import BaseModel
from fastapi import APIRouter
import base64, io, time
import pandas as pd
import numpy as np

router = APIRouter()


class IngestArgs(BaseModel):
    csv_b64: str


@router.post("/tools/ingest")
def ingest(args: IngestArgs) -> Dict[str, Any]:
    t0 = time.perf_counter()
    raw = base64.b64decode(args.csv_b64)
    df = pd.read_csv(io.BytesIO(raw))
    cols = []
    for c in df.columns:
        t = "number" if np.issubdtype(df[c].dtype, np.number) else "string"
        cols.append({"name": str(c), "type": t, "sample": df[c].head(3).tolist()})
    numeric = [c["name"] for c in cols if c["type"] == "number"]
    out = {
        "tool_version": "ingest/0.1.0",
        "columns": cols,
        "numeric_candidates": numeric,
        "preview_rows": df.head(5).to_dict(orient="records"),
        "runtime_ms": (time.perf_counter() - t0) * 1000,
    }
    return out


