# tests/unit/test_utils.py
from src.utils import simulate_progress

def test_simulate_progress_runs_without_errors():
    """Verifica que simulate_progress se ejecuta sin lanzar errores."""
    result = simulate_progress("Test task")
    assert result is None  # La función no devuelve nada

def test_simulate_progress_handles_empty_name():
    """Verifica que no falla si el nombre de tarea es vacío."""
    result = simulate_progress("")
    assert result is None