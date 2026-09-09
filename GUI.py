import sys
from pathlib import Path
import joblib
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QDoubleSpinBox,
    QComboBox, QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QMessageBox, QFrame
)
from PySide6.QtCore import Qt

def resource_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / filename
    return Path(__file__).parent / filename

MODEL_PATH = resource_path("SFCB_CatBoost_τ_Model.pkl")
model = joblib.load(MODEL_PATH)

ENV_MAP = {"Dry": 0, "Wet–dry": 1, "Seawater": 2}
FT_MAP = {"BFRP": 1, "GFRP": 2, "CFRP": 3}
CT_MAP = {"NC": 1, "SWSSC": 2, "CAC": 3, "ECC": 4}

TRAIN_RANGES = {
    "T": (20.0, 60.0),
    "t": (0.0, 630.0),
    "ηL": (0.01, 0.11),
    "d": (10.0, 25.5),
    "ρ": (0.16, 0.69),
    "fy": (150.10, 344.80),
    "fu": (406.66, 1144.00),
    "E1": (49.20, 155.00),
    "α": (0.05, 0.65),
    "fc": (17.60, 144.60),
    "l/d": (2.50, 20.00),
    "c/d": (2.28, 7.80)
}

class SFCBApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SFCB–Concrete Bond Strength Prediction System")
        self.resize(1260, 780)
        self.init_ui()
        self.apply_style()

    def create_spinbox(self, minimum, maximum, value, step=1.0, decimals=2):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setMinimumHeight(36)
        return box

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(24, 18, 24, 18)
        root_layout.setSpacing(14)

        # Header
        title = QLabel("SFCB–Concrete Bond Strength Prediction System")
        title.setObjectName("MainTitle")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("BO-CatBoost model for SFCB–concrete bond strength prediction")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("Separator")
        root_layout.addWidget(separator)

        # Main content layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(14)

        # Environmental parameters
        env_group = QGroupBox("Environmental parameters")
        env_layout = QGridLayout()
        env_layout.setHorizontalSpacing(14)
        env_layout.setVerticalSpacing(8)
        self.T = self.create_spinbox(-20, 100, 20, 1, 1)
        self.Env = QComboBox()
        self.Env.addItems(["Dry", "Wet–dry", "Seawater"])
        self.Env.setMinimumHeight(36)
        self.t = self.create_spinbox(0, 3000, 0, 1, 0)
        env_layout.addWidget(QLabel("Temperature T (°C)"), 0, 0)
        env_layout.addWidget(QLabel("Exposure environment Env"), 0, 1)
        env_layout.addWidget(QLabel("Exposure duration t (days)"), 0, 2)
        env_layout.addWidget(self.T, 1, 0)
        env_layout.addWidget(self.Env, 1, 1)
        env_layout.addWidget(self.t, 1, 2)
        env_group.setLayout(env_layout)

        # Geometric parameters
        geo_group = QGroupBox("Geometric parameters")
        geo_layout = QGridLayout()
        geo_layout.setHorizontalSpacing(14)
        geo_layout.setVerticalSpacing(8)
        self.eta_L = self.create_spinbox(0.0, 0.30, 0.05, 0.001, 3)
        self.d = self.create_spinbox(1, 100, 12, 0.1, 1)
        self.ld = self.create_spinbox(0.1, 50, 5, 0.1, 2)
        self.cd = self.create_spinbox(0.1, 20, 4.5, 0.1, 2)
        geo_labels = ["Modified concrete lug ratio ηL", "Bar diameter d (mm)", "Bond length-to-diameter ratio l/d", "Cover-to-diameter ratio c/d"]
        geo_widgets = [self.eta_L, self.d, self.ld, self.cd]
        for i in range(4):
            geo_layout.addWidget(QLabel(geo_labels[i]), 0, i)
            geo_layout.addWidget(geo_widgets[i], 1, i)
        geo_group.setLayout(geo_layout)

        # Material parameters
        mat_group = QGroupBox("Material parameters")
        mat_layout = QGridLayout()
        mat_layout.setHorizontalSpacing(14)
        mat_layout.setVerticalSpacing(8)
        self.FT = QComboBox()
        self.FT.addItems(["BFRP", "GFRP", "CFRP"])
        self.FT.setMinimumHeight(36)
        self.rho = self.create_spinbox(0.01, 0.99, 0.39, 0.01, 2)
        self.fy = self.create_spinbox(50, 1000, 250, 1, 2)
        self.fu = self.create_spinbox(100, 3000, 680, 1, 2)
        self.E1 = self.create_spinbox(1, 500, 100, 1, 2)
        self.alpha = self.create_spinbox(0.0, 1.5, 0.27, 0.01, 2)
        self.CT = QComboBox()
        self.CT.addItems(["NC", "SWSSC", "CAC", "ECC"])
        self.CT.setMinimumHeight(36)
        self.fc = self.create_spinbox(5, 300, 40, 1, 2)

        material_items = [
            ("FRP type FT", self.FT),
            ("Steel ratio ρ", self.rho),
            ("Yield strength fy (MPa)", self.fy),
            ("Ultimate strength fu (MPa)", self.fu),
            ("Initial elastic modulus E1 (GPa)", self.E1),
            ("Post-yield stiffness ratio α", self.alpha),
            ("Concrete type CT", self.CT),
            ("Concrete compressive strength fc (MPa)", self.fc)
        ]
        positions = [(0,0), (0,1), (0,2), (2,0), (2,1), (2,2), (4,0), (4,1)]
        for (label_text, widget), (row, col) in zip(material_items, positions):
            mat_layout.addWidget(QLabel(label_text), row, col)
            mat_layout.addWidget(widget, row+1, col)
        mat_group.setLayout(mat_layout)

        # Add parameter groups
        left_layout.addWidget(env_group)
        left_layout.addWidget(geo_group)
        left_layout.addWidget(mat_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        self.predict_button = QPushButton("Predict")
        self.predict_button.setObjectName("PredictButton")
        self.predict_button.setMinimumHeight(46)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("ResetButton")
        self.reset_button.setMinimumHeight(46)
        self.predict_button.clicked.connect(self.predict)
        self.reset_button.clicked.connect(self.reset_inputs)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.reset_button)
        left_layout.addLayout(button_layout)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(16)

        # Prediction result
        result_group = QGroupBox("Prediction result")
        result_layout = QVBoxLayout()
        result_caption = QLabel("Predicted bond strength")
        result_caption.setAlignment(Qt.AlignCenter)
        result_caption.setObjectName("ResultCaption")
        self.result_value = QLabel("--")
        self.result_value.setAlignment(Qt.AlignCenter)
        self.result_value.setObjectName("ResultValue")
        result_unit = QLabel("MPa")
        result_unit.setAlignment(Qt.AlignCenter)
        result_unit.setObjectName("ResultUnit")
        result_layout.addStretch()
        result_layout.addWidget(result_caption)
        result_layout.addWidget(self.result_value)
        result_layout.addWidget(result_unit)
        result_layout.addStretch()
        result_group.setLayout(result_layout)
        result_group.setMinimumHeight(210)

        # Prediction applicability
        status_group = QGroupBox("Prediction applicability")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("No prediction has been performed.")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("StatusNeutral")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)

        # Model information
        model_group = QGroupBox("Model information")
        model_layout = QVBoxLayout()
        model_text = QLabel(
            "Model: BO-CatBoost\n"
            "Output: Bond strength τ\n"
            "Input features: 15\n"
            "Database size: 360 specimens\n"
            "Testing R²: 0.9620\n"
            "Testing RMSE: 1.2878 MPa\n"
            "Testing MAE: 0.8944 MPa"
        )
        model_text.setWordWrap(True)
        model_layout.addWidget(model_text)
        model_group.setLayout(model_layout)

        # Notes
        note_group = QGroupBox("Notes")
        note_layout = QVBoxLayout()
        note_text = QLabel(
            "Predictions within the training-data ranges "
            "represent interpolation. Predictions outside these "
            "ranges involve extrapolation and should be interpreted "
            "with caution."
        )
        note_text.setWordWrap(True)
        note_layout.addWidget(note_text)
        note_group.setLayout(note_layout)

        # Add right panel
        right_layout.addWidget(result_group)
        right_layout.addWidget(status_group)
        right_layout.addWidget(model_group)
        right_layout.addWidget(note_group)
        right_layout.addStretch()

        # Main content proportions
        content_layout.addWidget(left_panel, stretch=3)
        content_layout.addWidget(right_panel, stretch=2)
        root_layout.addLayout(content_layout)

    def check_ranges(self, input_df):
        warnings = []
        for feature, (lower, upper) in TRAIN_RANGES.items():
            value = float(input_df.iloc[0][feature])
            if value < lower or value > upper:
                warnings.append(f"{feature} = {value:.3g} (training range: {lower:g}–{upper:g})")
        return warnings

    def predict(self):
        try:
            Env = ENV_MAP[self.Env.currentText()]
            FT = FT_MAP[self.FT.currentText()]
            CT = CT_MAP[self.CT.currentText()]
            input_df = pd.DataFrame([{
                "Env": Env, "CT": CT, "FT": FT,
                "T": self.T.value(), "t": self.t.value(),
                "ηL": self.eta_L.value(), "d": self.d.value(),
                "ρ": self.rho.value(), "fy": self.fy.value(),
                "fu": self.fu.value(), "E1": self.E1.value(),
                "α": self.alpha.value(), "fc": self.fc.value(),
                "l/d": self.ld.value(), "c/d": self.cd.value()
            }])
            input_df = input_df[model.feature_names_]
            warnings = self.check_ranges(input_df)
            prediction = model.predict(input_df)[0]
            self.result_value.setText(f"{prediction:.2f}")

            if len(warnings) == 0:
                self.status_label.setText(
                    "✓ All continuous input variables are within the training-data ranges.\n\n"
                    "The prediction represents interpolation within the model applicability domain."
                )
                self.status_label.setObjectName("StatusNormal")
            else:
                message = "⚠ Extrapolation warning\n\nThe following variable(s) are outside the training-data ranges:\n\n"
                message += "\n".join("• " + item for item in warnings)
                message += "\n\nThe prediction involves extrapolation and should be interpreted with caution."
                self.status_label.setText(message)
                self.status_label.setObjectName("StatusWarning")

            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)

        except Exception as error:
            QMessageBox.critical(self, "Prediction error", str(error))

    def reset_inputs(self):
        self.T.setValue(20)
        self.Env.setCurrentIndex(0)
        self.t.setValue(0)
        self.eta_L.setValue(0.05)
        self.d.setValue(12)
        self.ld.setValue(5)
        self.cd.setValue(4.5)
        self.FT.setCurrentIndex(0)
        self.rho.setValue(0.39)
        self.fy.setValue(250)
        self.fu.setValue(680)
        self.E1.setValue(100)
        self.alpha.setValue(0.27)
        self.CT.setCurrentIndex(0)
        self.fc.setValue(40)
        self.result_value.setText("--")
        self.status_label.setText("No prediction has been performed.")
        self.status_label.setObjectName("StatusNeutral")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #F4F7FA; }
            QLabel { font-size: 13px; color: #263238; }
            #MainTitle { font-size: 24px; font-weight: 700; color: #17365D; padding-top: 4px; }
            #Subtitle { font-size: 13px; color: #607D8B; padding-bottom: 6px; }
            #Separator { color: #D8E1E8; }
            QGroupBox {
                font-size: 14px; font-weight: 600; color: #17365D;
                border: 1px solid #CCD8E2; border-radius: 8px;
                margin-top: 12px; padding-top: 12px; background: white;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QDoubleSpinBox, QComboBox {
                background: #FFFFFF; border: 1px solid #B8C7D1; border-radius: 5px;
                padding: 5px 8px; font-size: 13px;
            }
            QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #3977A8; }
            QPushButton { border-radius: 6px; font-size: 14px; font-weight: 600; }
            #PredictButton { background: #245D87; color: white; border: none; }
            #PredictButton:hover { background: #1E4F74; }
            #ResetButton { background: white; color: #455A64; border: 1px solid #B8C7D1; }
            #ResetButton:hover { background: #EEF3F6; }
            #ResultCaption { font-size: 15px; color: #607D8B; }
            #ResultValue { font-size: 42px; font-weight: 700; color: #1E5C8A; padding-top: 12px; padding-bottom: 4px; }
            #ResultUnit { font-size: 16px; font-weight: 600; color: #546E7A; }
            #StatusNeutral {
                background: #F5F7F8; color: #607D8B; border: 1px solid #D8E0E5;
                border-radius: 6px; padding: 12px; font-size: 13px;
            }
            #StatusNormal {
                background: #EAF6EE; color: #246B3F; border: 1px solid #A8D5B5;
                border-radius: 6px; padding: 12px; font-size: 13px;
            }
            #StatusWarning {
                background: #FFF4E5; color: #8A5200; border: 1px solid #E8C483;
                border-radius: 6px; padding: 12px; font-size: 13px;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SFCBApp()
    window.show()
    sys.exit(app.exec())