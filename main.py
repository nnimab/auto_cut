import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QProgressBar, QSpinBox,
                           QListWidget, QGroupBox, QSlider, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import logging
from video_processor import process_video
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict

# 設置日誌
logging.basicConfig(level=logging.INFO)

@dataclass
class ProcessingParams:
    """影片處理參數"""
    clip_duration: int = 60  # 片段長度（秒）
    num_clips: int = 5      # 選取片段數量
    motion_threshold: float = 0.3  # 動作檢測閾值
    quality_preset: str = 'veryfast'  # 編碼預設值

class VideoProcessor(QThread):
    progress_updated = pyqtSignal(str, int)  # 檔案路徑, 進度
    processing_finished = pyqtSignal(str, str)  # 檔案路徑, 訊息
    
    def __init__(self, input_path: str, params: ProcessingParams):
        super().__init__()
        self.input_path = input_path
        self.output_path = self._get_output_path(input_path)
        self.params = params
        
    def _get_output_path(self, input_path):
        """生成輸出檔案路徑"""
        return input_path
    
    def run(self):
        """執行影片處理"""
        try:
            success = process_video(
                self.input_path,
                self.output_path,
                lambda p: self.progress_updated.emit(self.input_path, p),
                self.params
            )
            
            if success:
                output_dir = self.output_path.replace(".mp4", "_clips")
                self.processing_finished.emit(
                    self.input_path,
                    f"處理完成！\n輸出資料夾：{output_dir}"
                )
            else:
                self.processing_finished.emit(
                    self.input_path,
                    "處理失敗，請查看日誌了解詳情。"
                )
                
        except Exception as e:
            logging.error(f"處理影片時發生錯誤: {str(e)}")
            self.processing_finished.emit(self.input_path, f"處理失敗：{str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自動影片剪輯工具")
        self.setMinimumSize(800, 600)
        
        # 主要佈局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)
        
        # 左側面板（檔案列表和控制按鈕）
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # 檔案列表
        self.file_list = QListWidget()
        left_layout.addWidget(QLabel("待處理檔案："))
        left_layout.addWidget(self.file_list)
        
        # 檔案控制按鈕
        file_buttons = QHBoxLayout()
        self.add_btn = QPushButton("添加檔案")
        self.add_btn.clicked.connect(self.add_files)
        self.remove_btn = QPushButton("移除選中")
        self.remove_btn.clicked.connect(self.remove_selected)
        file_buttons.addWidget(self.add_btn)
        file_buttons.addWidget(self.remove_btn)
        left_layout.addLayout(file_buttons)
        
        layout.addWidget(left_panel)
        
        # 右側面板（參數設置和狀態顯示）
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # 參數設置組
        params_group = QGroupBox("參數設置")
        params_layout = QVBoxLayout()
        
        # 片段長度設置
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("片段長度（秒）："))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 300)
        self.duration_spin.setValue(60)
        duration_layout.addWidget(self.duration_spin)
        params_layout.addLayout(duration_layout)
        
        # 片段數量設置
        clips_layout = QHBoxLayout()
        clips_layout.addWidget(QLabel("選取片段數："))
        self.clips_spin = QSpinBox()
        self.clips_spin.setRange(1, 20)
        self.clips_spin.setValue(5)
        clips_layout.addWidget(self.clips_spin)
        params_layout.addLayout(clips_layout)
        
        # 動作閾值設置
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("動作閾值："))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(30)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.30")
        threshold_layout.addWidget(self.threshold_label)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v/100:.2f}")
        )
        params_layout.addLayout(threshold_layout)
        
        # 編碼預設值
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("編碼預設："))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['ultrafast', 'veryfast', 'faster', 'fast', 'medium'])
        self.preset_combo.setCurrentText('veryfast')
        preset_layout.addWidget(self.preset_combo)
        params_layout.addLayout(preset_layout)
        
        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)
        
        # 進度顯示區域
        self.progress_group = QGroupBox("處理進度")
        progress_layout = QVBoxLayout()
        self.progress_widgets: Dict[str, QProgressBar] = {}
        self.status_widgets: Dict[str, QLabel] = {}
        self.progress_group.setLayout(progress_layout)
        right_layout.addWidget(self.progress_group)
        
        # 開始處理按鈕
        self.process_btn = QPushButton("開始處理")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        right_layout.addWidget(self.process_btn)
        
        layout.addWidget(right_panel)
        
        # 初始化處理器
        self.processors: Dict[str, VideoProcessor] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=5)  # 最多同時處理5個檔案
    
    def add_files(self):
        """添加檔案"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇影片檔案",
            "",
            "影片檔案 (*.mp4 *.avi *.mov *.mkv)"
        )
        
        for path in file_paths:
            if path not in [self.file_list.item(i).text() for i in range(self.file_list.count())]:
                self.file_list.addItem(path)
                
                # 為每個檔案創建進度條和狀態標籤
                progress_layout = self.progress_group.layout()
                
                file_widget = QWidget()
                file_layout = QVBoxLayout()
                file_widget.setLayout(file_layout)
                
                # 檔案名標籤
                file_layout.addWidget(QLabel(os.path.basename(path)))
                
                # 進度條
                progress_bar = QProgressBar()
                self.progress_widgets[path] = progress_bar
                file_layout.addWidget(progress_bar)
                
                # 狀態標籤
                status_label = QLabel("就緒")
                self.status_widgets[path] = status_label
                file_layout.addWidget(status_label)
                
                progress_layout.addWidget(file_widget)
        
        self.process_btn.setEnabled(self.file_list.count() > 0)
    
    def remove_selected(self):
        """移除選中的檔案"""
        for item in self.file_list.selectedItems():
            path = item.text()
            # 移除進度條和狀態標籤
            if path in self.progress_widgets:
                self.progress_widgets[path].parent().deleteLater()
                del self.progress_widgets[path]
                del self.status_widgets[path]
            self.file_list.takeItem(self.file_list.row(item))
        
        self.process_btn.setEnabled(self.file_list.count() > 0)
    
    def get_processing_params(self) -> ProcessingParams:
        """獲取當前的處理參數"""
        return ProcessingParams(
            clip_duration=self.duration_spin.value(),
            num_clips=self.clips_spin.value(),
            motion_threshold=self.threshold_slider.value() / 100,
            quality_preset=self.preset_combo.currentText()
        )
    
    def start_processing(self):
        """開始處理所有檔案"""
        self.process_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        
        params = self.get_processing_params()
        
        # 為每個檔案創建處理器
        for i in range(self.file_list.count()):
            path = self.file_list.item(i).text()
            processor = VideoProcessor(path, params)
            processor.progress_updated.connect(self.update_progress)
            processor.processing_finished.connect(self.processing_done)
            self.processors[path] = processor
            processor.start()
    
    def update_progress(self, file_path: str, value: int):
        """更新進度條"""
        if file_path in self.progress_widgets:
            self.progress_widgets[file_path].setValue(value)
    
    def processing_done(self, file_path: str, message: str):
        """處理完成的回調"""
        if file_path in self.status_widgets:
            self.status_widgets[file_path].setText(message)
        
        # 檢查是否所有檔案都處理完成
        all_done = True
        for processor in self.processors.values():
            if processor.isRunning():
                all_done = False
                break
        
        if all_done:
            self.process_btn.setEnabled(True)
            self.add_btn.setEnabled(True)
            self.remove_btn.setEnabled(True)
            self.processors.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 