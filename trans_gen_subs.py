import os
import urllib.request
import zipfile
import subprocess
import sys
import threading
import time
import shutil

# Load torch to memory BEFORE PyQt5 to avoid a known DLL initialization conflict (WinError 1114)
try:
    import torch
except ImportError:
    pass

# Automatically install PyQt5 if missing to provide drag and drop GUI
try:
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog, QCheckBox
    from PyQt5.QtCore import pyqtSignal, QObject, Qt, QMetaObject, Q_ARG
    from PyQt5.QtGui import QFont, QColor
except ImportError:
    print("PyQt5 not found. Installing PyQt5 for GUI drag and drop...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog, QCheckBox
    from PyQt5.QtCore import pyqtSignal, QObject, Qt, QMetaObject, Q_ARG
    from PyQt5.QtGui import QFont, QColor

def get_bundle_dir():
    """Returns the base directory for resources. Works for both source and PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.abspath(".")

def download_ffmpeg():
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    zip_path = "ffmpeg.zip"
    temp_dir = "ffmpeg_temp"
    
    if os.path.exists("ffmpeg.exe"):
        try:
            # Check if the file is older than 30 days
            file_age_days = (time.time() - os.path.getmtime("ffmpeg.exe")) / (60 * 60 * 24)
            if file_age_days < 30:
                print(f"ffmpeg.exe is up to date ({int(file_age_days)} days old).")
                return
            else:
                print("ffmpeg.exe is more than 30 days old. Checking for update...")
                # Try to remove old version to force download
                os.remove("ffmpeg.exe")
        except PermissionError:
            print("Notice: ffmpeg.exe is currently in use or protected. Skipping update.")
            return
        except Exception as e:
            print(f"Notice: Could not check/remove old ffmpeg.exe ({e}). Using existing version.")
            return

    print("Downloading/Updating ffmpeg.exe (this ensures you have the latest codecs)...")
    try:
        # Step 1: Download
        urllib.request.urlretrieve(url, zip_path)
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
            raise Exception("Downloaded zip file is missing or too small (possible network error).")
        
        # Step 2: Extract
        print("Extracting ffmpeg...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test zip integrity
            zip_ref.testzip()
            zip_ref.extractall(temp_dir)
        
        # Step 3: Find and Move exe
        exe_path = None
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file == "ffmpeg.exe":
                    exe_path = os.path.join(root, file)
                    break
        
        if exe_path:
            shutil.move(exe_path, "ffmpeg.exe")
            print("ffmpeg.exe ready.")
        else:
            print("Failed to find ffmpeg.exe in the downloaded archive.")

    except zipfile.BadZipFile:
        print("Error: The downloaded ffmpeg.zip is corrupt. Please try running the script again.")
    except PermissionError as e:
        print(f"Permission Error: {e}. Try running the script as Administrator or check folder permissions.")
    except Exception as e:
        print(f"An error occurred while setting up FFmpeg: {e}")
    finally:
        # Cleanup
        if os.path.exists(zip_path):
            try: os.remove(zip_path)
            except: pass
        if os.path.exists(temp_dir):
            try: shutil.rmtree(temp_dir)
            except: pass

def run_transcription(video_file, model_name="medium", output_dir=".", use_fp16=False):
    download_ffmpeg()
    
    # Add current directory AND bundle directory to PATH so whisper can find ffmpeg.exe
    bundle_dir = get_bundle_dir()
    cwd_dir = os.path.abspath(".")
    
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if cwd_dir not in paths:
        paths.append(cwd_dir)
    if bundle_dir not in paths:
        paths.append(bundle_dir)
        
    os.environ["PATH"] = os.pathsep.join(paths)

    try:
        import whisper
        from whisper.utils import get_writer
    except ImportError:
        print("Installing openai-whisper...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        import whisper
        from whisper.utils import get_writer

    # Check for CUDA support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and use_fp16:
        print("\n[!] WARNING: FP16 is not supported on CPU. Using FP32 instead.")
        print("[!] To use GPU/FP16, install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        use_fp16 = False

    print(f"Target Device: {device.upper()}")
    print(f"Loading {model_name} Whisper model...")
    # 'base' model is faster but less accurate. 'medium' or 'large' offers much better translations at the cost of speed/memory.
    model = whisper.load_model(model_name, device=device)

    print(f"Transcribing and translating:\n{video_file}...")
    # task='translate' translates from source audio to English
    # verbose=True automatically prints translation lines step-by-step
    result = model.transcribe(video_file, language='ja', task='translate', verbose=True, fp16=use_fp16)

    print("Writing subtitle file...")
    writer = get_writer("srt", output_dir)
    writer(result, video_file, {})

    print(f"\nDone! Subtitles saved to {output_dir}")

# Setup stdout redirection so print() feeds into the GUI
class StreamProxy(QObject):
    new_text = pyqtSignal(str)
    
    def write(self, text):
        self.new_text.emit(str(text))
        
    def flush(self):
        pass

class DragDropApp(QWidget):
    job_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Subtitle Generator")
        self.resize(750, 600)
        self.setAcceptDrops(True)
        self.setStyleSheet("background-color: #f8f9fa;") # Light background for the window
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # --- 1. Model Selection ---
        model_layout = QHBoxLayout()
        model_layout.setSpacing(10)
        
        self.model_label = QLabel("🗣️👂 Openai Whisper model :")
        self.model_label.setFont(QFont("Arial", 11))
        self.model_label.setStyleSheet("color: #333;")
        model_layout.addWidget(self.model_label)
        
        model_combo_layout = QVBoxLayout()
        model_combo_layout.setSpacing(2)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
        self.model_combo.setCurrentText("medium")
        self.model_combo.setStyleSheet("padding: 6px; border: 1px solid #cce0ff; border-radius: 5px; background-color: #e6f2ff; color: #333;")
        model_combo_layout.addWidget(self.model_combo)
        
        self.model_subText = QLabel("Select the Whisper model for transcription. (openai/whisper)")
        self.model_subText.setFont(QFont("Arial", 8))
        self.model_subText.setStyleSheet("color: #666;")
        model_combo_layout.addWidget(self.model_subText)
        
        model_layout.addLayout(model_combo_layout)
        model_layout.setStretch(1, 1)
        main_layout.addLayout(model_layout)
        
        # Precision Dropdown
        precision_layout = QHBoxLayout()
        precision_layout.setSpacing(10)
        
        self.precision_label = QLabel("⚙️ Precision Mode :")
        self.precision_label.setFont(QFont("Arial", 11))
        self.precision_label.setStyleSheet("color: #333;")
        precision_layout.addWidget(self.precision_label)
        
        precision_combo_layout = QVBoxLayout()
        precision_combo_layout.setSpacing(2)
        
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["FP32", "FP16"])
        self.precision_combo.setCurrentText("FP32")
        self.precision_combo.setStyleSheet("padding: 6px; border: 1px solid #cce0ff; border-radius: 5px; background-color: #e6f2ff; color: #333;")
        precision_combo_layout.addWidget(self.precision_combo)
        
        self.precision_subText = QLabel("FP32 uses CPU (Better accuracy, slower). FP16 uses GPU (Slightly less accuracy, much faster).")
        self.precision_subText.setFont(QFont("Arial", 8))
        self.precision_subText.setStyleSheet("color: #666;")
        precision_combo_layout.addWidget(self.precision_subText)
        
        precision_layout.addLayout(precision_combo_layout)
        precision_layout.setStretch(1, 1)
        main_layout.addLayout(precision_layout)
        
        # --- Hardware Status Indicator ---
        self.hw_layout = QHBoxLayout()
        cuda_ready = False
        try:
            import torch
            cuda_ready = torch.cuda.is_available()
        except:
            pass
            
        status_text = "🟢 Hardware Acceleration (CUDA): Available" if cuda_ready else "🔴 Hardware Acceleration (CUDA): NOT FOUND (Using CPU)"
        status_color = "#28a745" if cuda_ready else "#dc3545"
        
        self.hw_label = QLabel(status_text)
        self.hw_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.hw_label.setStyleSheet(f"color: {status_color}; padding: 5px; background-color: {status_color}22; border-radius: 4px;")
        self.hw_layout.addWidget(self.hw_label)
        main_layout.addLayout(self.hw_layout)
        
        # --- 2. Drag & Drop Area ---
        self.drop_label = QLabel("📄 Drag & Drop a Video File Here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.drop_label.setStyleSheet("padding: 40px; border: 2px dashed #aaa; background-color: #fdfdfd; border-radius: 8px; color: #333;")
        main_layout.addWidget(self.drop_label)
        
        # --- 3. File Location Row ---
        location_layout = QHBoxLayout()
        location_layout.setSpacing(10)
        
        self.loc_icon = QLabel("📁📍")
        self.loc_icon.setFont(QFont("Arial", 12))
        location_layout.addWidget(self.loc_icon)
        
        self.loc_input = QLineEdit()
        self.loc_input.setPlaceholderText("")
        self.loc_input.setStyleSheet("padding: 8px; border: 1px solid #cce0ff; border-radius: 5px; background-color: #e6f2ff; color: #333;")
        location_layout.addWidget(self.loc_input)
        
        self.browse_btn = QPushButton("BROWSE")
        self.browse_btn.setStyleSheet("padding: 8px 15px; color: #555; border: 1px solid #ccc; border-radius: 5px; background-color: #f0f0f0;")
        self.browse_btn.clicked.connect(self.browse_folder)
        location_layout.addWidget(self.browse_btn)
        
        main_layout.addLayout(location_layout)
        
        # --- 4. Start Button ---
        self.start_btn = QPushButton("Start Subtitles Generation")
        self.start_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.start_btn.setStyleSheet("padding: 12px; background-color: #cccccc; color: white; border: none; border-radius: 6px;")
        self.start_btn.setEnabled(False) 
        self.start_btn.clicked.connect(self.start_generation)
        main_layout.addWidget(self.start_btn)
        
        # --- 5. Terminal Console ---
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
        self.text_edit.setStyleSheet("background-color: #1e1e19; color: #d4d4af; padding: 10px; border-radius: 5px; border: none;")
        main_layout.addWidget(self.text_edit)
        
        self.setLayout(main_layout)
        
        self.target_video_file = None
        
        # Redirect standard output and error
        self.stream = StreamProxy()
        self.stream.new_text.connect(self.append_text)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stream
        sys.stderr = self.stream
        
        self.job_finished.connect(self.on_job_finished)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet("padding: 40px; border: 2px dashed #0078D7; background-color: #e0f0ff; border-radius: 8px; color: #333;")

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet("padding: 40px; border: 2px dashed #aaa; background-color: #fdfdfd; border-radius: 8px; color: #333;")

    def dropEvent(self, event):
        self.drop_label.setStyleSheet("padding: 40px; border: 2px dashed #aaa; background-color: #fdfdfd; border-radius: 8px; color: #333;")
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.target_video_file = file_path
            self.drop_label.setText(f"📄 Selected:\n{os.path.basename(file_path)}")
            
            # Default output directory to file's folder
            self.loc_input.setText(os.path.dirname(file_path))
            
            # Enable button
            self.start_btn.setEnabled(True)
            self.start_btn.setStyleSheet("padding: 12px; background-color: #4CAF50; color: white; border: none; border-radius: 6px;")
            print(f"Loaded: {os.path.basename(file_path)}")
            print("Ready to Start.")

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.loc_input.setText(folder)

    def start_generation(self):
        if not self.target_video_file:
            return
            
        model_choice = self.model_combo.currentText()
        out_dir = self.loc_input.text()
        use_fp16 = self.precision_combo.currentText() == "FP16"
        
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Generating... (Check console below)")
        self.start_btn.setStyleSheet("padding: 12px; background-color: #aaaaaa; color: white; border: none; border-radius: 6px;")
        
        print(f"\n{'='*50}\nStarting: {os.path.basename(self.target_video_file)}\nModel: {model_choice}\nOutput: {out_dir}\nFP16: {use_fp16}\n{'='*50}\n")
        
        threading.Thread(target=self.process_file, args=(self.target_video_file, model_choice, out_dir, use_fp16), daemon=True).start()

    def process_file(self, file_path, model_choice, out_dir, use_fp16):
        try:
            run_transcription(file_path, model_choice, out_dir, use_fp16)
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.job_finished.emit()

    def on_job_finished(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Subtitles Generation")
        self.start_btn.setStyleSheet("padding: 12px; background-color: #4CAF50; color: white; border: none; border-radius: 6px;")
        self.drop_label.setText("📄 Drag & Drop another Video File Here")
        self.target_video_file = None

    def append_text(self, text):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
        
    def closeEvent(self, event):
        # Restore normal stdout when closing
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DragDropApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
