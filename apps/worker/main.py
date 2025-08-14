from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

# Import routers from tools package
from tools.ab_test_ttest import router as ab_router
from tools.markov_mcs import router as markov_router
from tools.plot_line import router as plot_line_router
from tools.plot_bar import router as plot_bar_router
from tools.plot_bar_with_ci import router as plot_bar_ci_router
from tools.power_curve import router as power_curve_router
from tools.causal_impact import router as did_router
from tools.forecast_arima import router as arima_router
from tools.summarize_results import router as summarize_router
from tools.ingest import router as ingest_router


app = FastAPI(title="Agent Tools")

# Static artifacts directory
os.makedirs('artifacts', exist_ok=True)
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")


@app.get("/health")
def health():
    return {"ok": True}


# Register all tool routers
app.include_router(ab_router)
app.include_router(markov_router)
app.include_router(plot_line_router)
app.include_router(plot_bar_router)
app.include_router(plot_bar_ci_router)
app.include_router(power_curve_router)
app.include_router(did_router)
app.include_router(arima_router)
app.include_router(summarize_router)
app.include_router(ingest_router)


