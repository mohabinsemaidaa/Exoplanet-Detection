from streamlit.testing.v1 import AppTest


def _run_app():
    at = AppTest.from_file("app/dashboard.py")
    at.run(timeout=30)
    return at


def test_all_tabs_render_without_error():
    at = _run_app()
    assert not at.exception
    assert len(at.tabs) == 4


def test_prediction_button_returns_a_result():
    at = _run_app()
    at.button[0].click().run(timeout=30)

    assert not at.exception
    result_messages = [m.value for m in at.success] + [m.value for m in at.warning] + [
        m.value for m in at.error
    ]
    assert any("Predicted disposition" in message for message in result_messages)
