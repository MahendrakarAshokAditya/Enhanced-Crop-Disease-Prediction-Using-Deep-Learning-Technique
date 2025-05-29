import sys
import cv2
import qtawesome as qta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from model_handler import DiseaseDetector
from PIL import Image
import io
import random

class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setFixedSize(700, 400)
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, 700, 400)
        gradient.setColorAt(0, QColor(34, 193, 195))
        gradient.setColorAt(1, QColor(253, 187, 45))
        
        pixmap = QPixmap(700, 400)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, 700, 400, gradient)
        
        # Add decorative elements
        pen = QPen(QColor(255, 255, 255, 100))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw some decorative circles
        for i in range(15):
            x = random.randint(0, 700)
            y = random.randint(0, 400)
            size = random.randint(10, 50)
            painter.drawEllipse(x, y, size, size)
        
        # Draw main text with shadow effect
        shadow_color = QColor(0, 0, 0, 80)
        painter.setPen(shadow_color)
        painter.setFont(QFont('Segoe UI', 28, QFont.Bold))
        painter.drawText(3, 153, 700, 50, Qt.AlignCenter, 'Enhancing Crop Disease Prediction')
        
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont('Segoe UI', 28, QFont.Bold))
        painter.drawText(0, 150, 700, 50, Qt.AlignCenter, 'Enhancing Crop Disease Prediction')
        
        painter.setPen(QColor(255, 255, 255, 200))
        painter.setFont(QFont('Segoe UI', 14))
        painter.drawText(0, 200, 700, 30, Qt.AlignCenter, 'Using Deep Learning for Plant Health Analysis')
        
        # Add a progress bar
        self.progress_rect = QRect(200, 280, 300, 8)
        painter.setBrush(QColor(255, 255, 255, 50))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.progress_rect, 4, 4)
        
        painter.end()
        self.setPixmap(pixmap)
        
        # Setup animation timer
        self.progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(30)
    
    def update_progress(self):
        self.progress += 1
        if self.progress > 100:
            self.timer.stop()
            return
            
        # Update progress bar
        pixmap = self.pixmap().copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw progress
        progress_width = int(self.progress_rect.width() * (self.progress / 100))
        progress_bar = QRect(self.progress_rect.x(), self.progress_rect.y(), 
                             progress_width, self.progress_rect.height())
        
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(progress_bar, 4, 4)
        painter.end()
        
        self.setPixmap(pixmap)

def run_app():
    app = QApplication(sys.argv)
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Process events to display splash screen
    app.processEvents()
    
    # Create and show main window
    window = MainWindow()
    window.setGeometry(100, 100, 1200, 800)
    window.show()
    
    # Close splash screen
    splash.close()
    
    sys.exit(app.exec_())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = DiseaseDetector()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Enhancing Crop Disease Prediction Using Deep Learning')
        self.setWindowIcon(qta.icon('fa5s.leaf', color='#27ae60'))
        
        # Set modern style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f0f2f5, stop:1 #e0e5ec);
            }
            QPushButton {
                background-color: #4a6fa5;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #5a7fb5;
            }
            QPushButton:pressed {
                background-color: #3a5f95;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #34495e;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: #e0e5ec;
            }
            QTextEdit {
                background-color: white;
                border-radius: 8px;
                padding: 10px;
                border: 2px solid #bdc3c7;
                font-family: 'Segoe UI', sans-serif;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #bdc3c7;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Add header with logo and title
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = self.create_logo_pixmap()
        logo_label.setPixmap(logo_pixmap)
        logo_label.setFixedSize(64, 64)
        header_layout.addWidget(logo_label)
        
        # Title and subtitle
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(10, 0, 0, 0)
        title_layout.setSpacing(0)
        
        title_label = QLabel('Crop Disease Detector')
        title_label.setStyleSheet('font-size: 24px; font-weight: bold; color: #2c3e50;')
        subtitle_label = QLabel('Using Deep Learning for Plant Health Analysis')
        subtitle_label.setStyleSheet('font-size: 14px; color: #7f8c8d;')
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        header_layout.addWidget(title_widget)
        header_layout.addStretch()
        
        main_layout.addWidget(header)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet('background-color: #bdc3c7;')
        main_layout.addWidget(separator)
        
        # Content area with left and right panels
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(20)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # Upload section
        upload_group = QGroupBox('Image Input')
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setSpacing(15)
        
        # Styled buttons with icons
        self.upload_btn = QPushButton('  Upload Image')
        self.upload_btn.setIcon(qta.icon('fa5s.file-upload', color='white'))
        self.upload_btn.setIconSize(QSize(20, 20))
        self.upload_btn.clicked.connect(self.upload_image)
        
        self.capture_btn = QPushButton('  Capture Image')
        self.capture_btn.setIcon(qta.icon('fa5s.camera', color='white'))
        self.capture_btn.setIconSize(QSize(20, 20))
        self.capture_btn.clicked.connect(self.capture_image)
        
        self.process_btn = QPushButton('  Detect Disease')
        self.process_btn.setIcon(qta.icon('fa5s.search', color='white'))
        self.process_btn.setIconSize(QSize(20, 20))
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
        """)
        
        upload_layout.addWidget(self.upload_btn)
        upload_layout.addWidget(self.capture_btn)
        upload_layout.addWidget(self.process_btn)
        
        # Add help section
        help_group = QGroupBox('How It Works')
        help_layout = QVBoxLayout(help_group)
        
        help_text = QLabel("""
        <ol>
          <li>Upload or capture an image of a plant leaf</li>
          <li>Click 'Detect Disease' to analyze</li>
          <li>View detailed results and recommendations</li>
        </ol>
        """)
        help_text.setTextFormat(Qt.RichText)
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        
        left_layout.addWidget(upload_group)
        left_layout.addWidget(help_group)
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Right panel for image and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        
        # Image display with card-like styling
        image_card = QWidget()
        image_card.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        """)
        image_card_layout = QVBoxLayout(image_card)
        
        image_header = QLabel('Plant Image')
        image_header.setStyleSheet('font-size: 16px; font-weight: bold; color: #34495e;')
        image_header.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(450, 350)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px dashed #3498db;
            border-radius: 8px;
            background-color: #f8f9fa;
            padding: 10px;
        """)
        
        # Add placeholder text and icon
        placeholder_layout = QVBoxLayout()
        placeholder_layout.setAlignment(Qt.AlignCenter)
        
        placeholder_icon = QLabel()
        placeholder_icon_pixmap = qta.icon('fa5s.leaf', color='#bdc3c7').pixmap(64, 64)
        placeholder_icon.setPixmap(placeholder_icon_pixmap)
        placeholder_icon.setAlignment(Qt.AlignCenter)
        
        placeholder_text = QLabel('No image selected\nUpload or capture an image to begin')
        placeholder_text.setStyleSheet('color: #7f8c8d; font-size: 14px;')
        placeholder_text.setAlignment(Qt.AlignCenter)
        
        placeholder_layout.addWidget(placeholder_icon)
        placeholder_layout.addWidget(placeholder_text)
        self.image_label.setLayout(placeholder_layout)
        
        image_card_layout.addWidget(image_header)
        image_card_layout.addWidget(self.image_label)
        
        # Results display with card-like styling
        results_card = QWidget()
        results_card.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        """)
        results_card_layout = QVBoxLayout(results_card)
        
        results_header = QLabel('Analysis Results')
        results_header.setStyleSheet('font-size: 16px; font-weight: bold; color: #34495e;')
        results_header.setAlignment(Qt.AlignCenter)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(200)
        self.result_text.setStyleSheet("""
            border: none;
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Segoe UI', sans-serif;
        """)
        
        # Add placeholder text
        self.result_text.setHtml("""
        <div style='color: #7f8c8d; text-align: center; margin-top: 50px;'>
            <p>No analysis results yet</p>
            <p>Upload an image and click 'Detect Disease' to analyze</p>
        </div>
        """)
        
        results_card_layout.addWidget(results_header)
        results_card_layout.addWidget(self.result_text)
        
        right_layout.addWidget(image_card)
        right_layout.addWidget(results_card)
        layout.addWidget(right_panel, 1)
        
        main_layout.addWidget(content, 1)
        
        # Add footer
        footer = QLabel('© 2025 Crop Disease Detector | AI-Powered Plant Health Analysis')
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet('color: #95a5a6; font-size: 12px; padding: 10px;')
        main_layout.addWidget(footer)
        
        self.setMinimumSize(1000, 800)
        self.center()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if file_name:
            self.load_image(file_name)
            
    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, 'Error', 'Could not access the camera.')
            return
            
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create temporary QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(img_byte_arr)
            self.display_image(pixmap)
            self.current_image = image
            self.process_btn.setEnabled(True)
            
    def load_image(self, file_path):
        image = Image.open(file_path)
        pixmap = QPixmap(file_path)
        self.display_image(pixmap)
        self.current_image = image
        self.process_btn.setEnabled(True)
        
    def display_image(self, pixmap):
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
    def process_image(self):
        if not hasattr(self, 'current_image'):
            return
            
        self.process_btn.setEnabled(False)
        self.result_text.clear()
        self.result_text.append('Processing image...')
        
        # Create a QThread for processing
        self.thread = QThread()
        self.worker = DetectionWorker(self.detector, self.current_image)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.result.connect(self.display_results)
        
        self.thread.start()
        
    def create_logo_pixmap(self):
        # Create a custom logo
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw leaf shape
        path = QPainterPath()
        path.moveTo(32, 8)
        path.cubicTo(45, 15, 60, 30, 50, 50)
        path.cubicTo(40, 60, 20, 60, 15, 50)
        path.cubicTo(10, 40, 20, 15, 32, 8)
        
        # Fill with gradient
        gradient = QLinearGradient(0, 0, 64, 64)
        gradient.setColorAt(0, QColor(46, 204, 113))
        gradient.setColorAt(1, QColor(26, 188, 156))
        
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(39, 174, 96), 2))
        painter.drawPath(path)
        
        # Draw scan line
        painter.setPen(QPen(QColor(255, 255, 255, 180), 2))
        painter.drawLine(20, 32, 44, 32)
        
        painter.end()
        return pixmap
        
    def display_image(self, pixmap):
        # Clear the placeholder layout if it exists
        if self.image_label.layout() is not None:
            # Remove the layout and its items
            while self.image_label.layout().count():
                item = self.image_label.layout().takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            QWidget().setLayout(self.image_label.layout())
            
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def display_results(self, result):
        self.result_text.clear()
        if result:
            confidence = result['confidence'] * 100
            confidence_level = 'High' if confidence > 80 else 'Medium' if confidence > 60 else 'Low'
            confidence_color = '#27ae60' if confidence > 80 else '#f39c12' if confidence > 60 else '#e74c3c'
            
            # Create a more visually appealing results display
            html = f"""
            <div style='font-family: "Segoe UI", sans-serif;'>
                <div style='background: linear-gradient(135deg, {confidence_color}20, {confidence_color}10); 
                            padding: 15px; border-radius: 8px; margin-bottom: 20px; 
                            border-left: 5px solid {confidence_color};'>
                    <h2 style='color: #2c3e50; margin-top: 0;'>{result['disease_name']}</h2>
                    <div style='display: flex; align-items: center;'>
                        <div style='background-color: {confidence_color}; 
                                    width: {confidence}%; height: 8px; 
                                    border-radius: 4px;'></div>
                        <span style='margin-left: 10px; color: {confidence_color}; 
                                    font-weight: bold;'>{confidence:.1f}% - {confidence_level} Confidence</span>
                    </div>
                </div>
                
                <div style='background-color: #f8f9fa; padding: 15px; 
                            border-radius: 8px; margin-bottom: 15px;'>
                    <h3 style='color: #2c3e50; margin-top: 0;'>Description</h3>
                    <p style='color: #34495e;'>{result['info']['description']}</p>
                </div>
                
                <div style='display: flex; gap: 15px; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 200px; background-color: #f8f9fa; 
                                padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                        <h3 style='color: #2c3e50; margin-top: 0;'>Symptoms</h3>
                        <ul style='color: #34495e; padding-left: 20px;'>
                        {''.join([f'<li>{s}</li>' for s in result['info']['symptoms']]) if result['info']['symptoms'] else '<li>No symptoms information available</li>'}
                        </ul>
                    </div>
                    
                    <div style='flex: 1; min-width: 200px; background-color: #f8f9fa; 
                                padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                        <h3 style='color: #2c3e50; margin-top: 0;'>Treatment</h3>
                        <ul style='color: #34495e; padding-left: 20px;'>
                        {''.join([f'<li>{t}</li>' for t in result['info']['treatment']]) if result['info']['treatment'] else '<li>No treatment information available</li>'}
                        </ul>
                    </div>
                </div>
                
                <div style='background-color: #f8f9fa; padding: 15px; 
                            border-radius: 8px; margin-bottom: 15px;'>
                    <h3 style='color: #2c3e50; margin-top: 0;'>Prevention</h3>
                    <ul style='color: #34495e; padding-left: 20px;'>
                    {''.join([f'<li>{p}</li>' for p in result['info']['prevention']]) if result['info']['prevention'] else '<li>No prevention information available</li>'}
                    </ul>
                </div>
            </div>
            """
            self.result_text.setHtml(html)
        else:
            error_html = """
            <div style='text-align: center; padding: 20px;'>
                <div style='font-size: 48px; color: #e74c3c;'>⚠️</div>
                <h3 style='color: #e74c3c;'>No Disease Detected</h3>
                <p style='color: #7f8c8d;'>The system couldn't detect any disease in the provided image or an error occurred during processing.</p>
                <p style='color: #7f8c8d;'>Try uploading a clearer image or ensure the leaf is properly visible.</p>
            </div>
            """
            self.result_text.setHtml(error_html)
            
        self.process_btn.setEnabled(True)

class DetectionWorker(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(dict)
    
    def __init__(self, detector, image):
        super().__init__()
        self.detector = detector
        self.image = image
        
    def run(self):
        result = self.detector.detect_disease(self.image)
        # Handle None result by emitting an empty dictionary
        self.result.emit(result if result is not None else {})
        self.finished.emit()

def main():
    app = QApplication(sys.argv)
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Process events for 3 seconds
    start = QTime.currentTime()
    while start.msecsTo(QTime.currentTime()) < 3000:
        app.processEvents()
    
    # Create and show main window
    window = MainWindow()
    window.show()
    splash.finish(window)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()