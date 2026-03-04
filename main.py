import sys

from app.main_window import MainWindow

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - runtime only
    raise SystemExit("PySide6 is required. Install with: pip install -r requirements.txt") from exc


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

