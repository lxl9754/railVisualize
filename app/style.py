def get_app_stylesheet() -> str:
    return """
    /* Base widgets */
    QWidget {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: "Microsoft YaHei", "Segoe UI";
        font-size: 13px;
    }

    /* Group boxes */
    QGroupBox {
        border: 1px solid #3f3f46;
        border-radius: 6px;
        margin-top: 15px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: #4daaf9;
    }

    /* Buttons */
    QPushButton {
        background-color: #333337;
        border: 1px solid #3f3f46;
        border-radius: 4px;
        padding: 6px 12px;
        color: #e2e2e2;
    }
    QPushButton:hover {
        background-color: #3e3e42;
        border: 1px solid #007acc;
    }
    QPushButton:pressed {
        background-color: #007acc;
        color: white;
    }

    /* Inputs */
    QLineEdit, QSpinBox, QDoubleSpinBox {
        background-color: #2d2d30;
        border: 1px solid #3f3f46;
        border-radius: 3px;
        padding: 3px;
        color: #d4d4d4;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #007acc;
    }

    /* Radio/checkbox */
    QRadioButton, QCheckBox {
        spacing: 8px;
    }
    QRadioButton::indicator, QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }

    /* Progress bar */
    QProgressBar {
        border: 1px solid #3f3f46;
        border-radius: 4px;
        text-align: center;
        background-color: #2d2d30;
    }
    QProgressBar::chunk {
        background-color: #007acc;
        border-radius: 3px;
    }
    """
