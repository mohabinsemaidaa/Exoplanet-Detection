from streamlit.testing.v1 import AppTest


def _run_app():
    at = AppTest.from_file("app/dashboard.py")
    at.run(timeout=15)
    return at


def test_overview_page_loads_by_default():
    at = _run_app()
    assert not at.exception


def test_dataset_exploration_page_loads():
    at = _run_app()
    at.sidebar.radio[0].set_value("Dataset Exploration").run(timeout=15)
    assert not at.exception


def test_model_performance_page_loads():
    at = _run_app()
    at.sidebar.radio[0].set_value("Model Performance").run(timeout=15)
    assert not at.exception


def test_prediction_page_loads_and_predicts():
    at = _run_app()
    at.sidebar.radio[0].set_value("Prediction").run(timeout=15)
    assert not at.exception

    at.button[0].click().run(timeout=15)
    assert not at.exception
