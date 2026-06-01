from aitviewer.viewer import Viewer, _resolve_qt_window_type


def test_resolve_qt_window_type_falls_back_to_pyqt5_when_pyqt6_is_missing():
    def find_spec(module_name):
        return object() if module_name.startswith("PyQt5") else None

    assert _resolve_qt_window_type("pyqt6", find_spec=find_spec) == "pyqt5"


def test_resolve_qt_window_type_keeps_pyqt6_when_no_qt_binding_is_available():
    def find_spec(module_name):
        if module_name.startswith("PyQt5"):
            raise ModuleNotFoundError(module_name)
        return None

    assert _resolve_qt_window_type("pyqt6", find_spec=find_spec) == "pyqt6"


def test_viewer_render_delegates_to_on_render():
    viewer = object.__new__(Viewer)
    calls = []

    def on_render(*args, **kwargs):
        calls.append((args, kwargs))

    viewer.on_render = on_render

    viewer.render(1.0, 0.5)

    assert calls == [((1.0, 0.5), {})]
