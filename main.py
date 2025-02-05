import sys
import os
import shutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QProgressBar,
                           QListWidget, QHBoxLayout, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QFormLayout, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import logging
from video_processor import process_video

# 設置日誌
logging.basicConfig(level=logging.INFO)

class VideoProcessor(QThread):
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(str)
    
    def __init__(self, input_path, clip_count=5, min_duration=10, max_duration=60):
        super().__init__()
        self.input_path = input_path
        self.output_path = self._get_output_path(input_path)
        self.clip_count = clip_count
        self.min_duration = min_duration
        self.max_duration = max_duration
        self._is_cancelled = False
        
    def _get_output_path(self, input_path):
        """生成輸出檔案路徑"""
        # 這裡返回的是原始檔案路徑，實際的輸出資料夾會在process_video中創建
        return input_path
    
    def cancel(self):
        """取消處理"""
        self._is_cancelled = True
        self.wait()  # 等待線程結束
        
        # 清理輸出資料夾
        output_dir = self.output_path.replace(".mp4", "_clips")
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                logging.info(f"已清理資料夾: {output_dir}")
            except Exception as e:
                logging.error(f"清理資料夾失敗: {str(e)}")
    
    def run(self):
        """執行影片處理"""
        try:
            success = process_video(
                self.input_path,
                self.output_path,
                self.progress_updated.emit,
                self.clip_count,
                self.min_duration,
                self.max_duration,
                lambda: self._is_cancelled  # 傳遞取消檢查函數
            )
            
            if self._is_cancelled:
                self.processing_finished.emit("處理已取消")
                return
            
            if success:
                output_dir = self.output_path.replace(".mp4", "_clips")
                self.processing_finished.emit(f"處理完成！\n輸出資料夾：{output_dir}")
            else:
                self.processing_finished.emit("處理失敗，請查看日誌了解詳情。")
                
        except Exception as e:
            logging.error(f"處理影片時發生錯誤: {str(e)}")
            self.processing_finished.emit(f"處理失敗：{str(e)}")

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
        
        # 左側面板 - 檔案列表和控制項
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # 檔案列表
        self.file_list = QListWidget()
        left_layout.addWidget(QLabel("待處理影片列表："))
        left_layout.addWidget(self.file_list)
        
        # 檔案控制按鈕
        file_buttons = QHBoxLayout()
        self.add_btn = QPushButton("添加影片")
        self.remove_btn = QPushButton("移除選中")
        self.add_btn.clicked.connect(self.add_files)
        self.remove_btn.clicked.connect(self.remove_selected)
        file_buttons.addWidget(self.add_btn)
        file_buttons.addWidget(self.remove_btn)
        left_layout.addLayout(file_buttons)
        
        # 參數設置組
        params_group = QGroupBox("參數設置")
        params_layout = QFormLayout()
        
        # 片段數量設置
        self.clip_count_spin = QSpinBox()
        self.clip_count_spin.setRange(1, 20)
        self.clip_count_spin.setValue(5)
        params_layout.addRow("剪輯片段數量：", self.clip_count_spin)
        
        # 時長範圍設置
        self.min_duration_spin = QSpinBox()
        self.min_duration_spin.setRange(1, 300)  # 擴大範圍
        self.min_duration_spin.setValue(10)
        self.min_duration_spin.valueChanged.connect(self.on_duration_changed)
        params_layout.addRow("最短片段時長（秒）：", self.min_duration_spin)
        
        self.max_duration_spin = QSpinBox()
        self.max_duration_spin.setRange(1, 300)  # 擴大範圍
        self.max_duration_spin.setValue(60)
        self.max_duration_spin.valueChanged.connect(self.on_duration_changed)
        params_layout.addRow("最長片段時長（秒）：", self.max_duration_spin)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # 開始處理按鈕
        self.process_btn = QPushButton("開始處理所有影片")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)
        
        layout.addWidget(left_panel)
        
        # 右側面板 - 進度顯示
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        self.progress_widgets = {}
        right_layout.addStretch()
        
        layout.addWidget(right_panel)
        
        # 設置左右比例為1:1
        layout.setStretch(0, 1)
        layout.setStretch(1, 1)
        
        self.processors = {}
        
        # 初始檢查時長設置
        self.on_duration_changed()
    
    def on_duration_changed(self):
        """當時長設置改變時進行驗證"""
        min_duration = self.min_duration_spin.value()
        max_duration = self.max_duration_spin.value()
        
        if min_duration > max_duration:
            QMessageBox.warning(
                self,
                "參數錯誤",
                "最短片段時長不能大於最長片段時長！",
                QMessageBox.StandardButton.Ok
            )
            # 還原之前的有效值
            if self.sender() == self.min_duration_spin:
                self.min_duration_spin.setValue(max_duration)
            else:
                self.max_duration_spin.setValue(min_duration)
            return
        
        # 更新最小值和最大值的範圍
        self.min_duration_spin.setMaximum(max_duration)
        self.max_duration_spin.setMinimum(min_duration)
    
    def add_files(self):
        """添加多個影片檔案"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇影片檔案",
            "",
            "影片檔案 (*.mp4 *.avi *.mov *.mkv)"
        )
        
        for file_path in files:
            if self.file_list.findItems(file_path, Qt.MatchFlag.MatchExactly):
                continue
            self.file_list.addItem(file_path)
        
        self.process_btn.setEnabled(self.file_list.count() > 0)
    
    def remove_selected(self):
        """移除選中的影片"""
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self.process_btn.setEnabled(self.file_list.count() > 0)
    
    def create_progress_widget(self, file_path):
        """為每個影片創建進度顯示組件"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 上方佈局：檔案名稱和取消按鈕
        top_layout = QHBoxLayout()
        
        # 檔案名稱標籤
        name_label = QLabel(os.path.basename(file_path))
        top_layout.addWidget(name_label)
        
        # 取消按鈕
        cancel_btn = QPushButton("取消")
        cancel_btn.setEnabled(False)  # 初始時禁用
        top_layout.addWidget(cancel_btn)
        
        layout.addLayout(top_layout)
        
        # 進度條
        progress_bar = QProgressBar()
        layout.addWidget(progress_bar)
        
        # 狀態標籤
        status_label = QLabel("等待處理...")
        layout.addWidget(status_label)
        
        return {
            'widget': widget,
            'progress_bar': progress_bar,
            'status_label': status_label,
            'cancel_btn': cancel_btn
        }
    
    def start_processing(self):
        """開始處理所有影片"""
        # 再次檢查參數有效性
        if self.min_duration_spin.value() > self.max_duration_spin.value():
            QMessageBox.warning(
                self,
                "參數錯誤",
                "最短片段時長不能大於最長片段時長！請調整參數後重試。",
                QMessageBox.StandardButton.Ok
            )
            return
            
        self.process_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        
        # 清理右側面板
        for widgets in self.progress_widgets.values():
            widgets['widget'].deleteLater()
        self.progress_widgets.clear()
        
        # 獲取所有檔案路徑
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        
        # 為每個檔案創建進度顯示
        right_panel = self.centralWidget().layout().itemAt(1).widget()
        right_layout = right_panel.layout()
        
        for file_path in files:
            widgets = self.create_progress_widget(file_path)
            self.progress_widgets[file_path] = widgets
            right_layout.insertWidget(right_layout.count() - 1, widgets['widget'])
            
            # 創建處理器
            processor = VideoProcessor(
                file_path,
                self.clip_count_spin.value(),
                self.min_duration_spin.value(),
                self.max_duration_spin.value()
            )
            
            # 設置取消按鈕功能
            cancel_btn = widgets['cancel_btn']
            cancel_btn.setEnabled(True)
            cancel_btn.clicked.connect(lambda checked, p=file_path: self.cancel_processing(p))
            
            processor.progress_updated.connect(
                lambda v, p=file_path: self.update_progress(p, v)
            )
            processor.processing_finished.connect(
                lambda m, p=file_path: self.processing_done(p, m)
            )
            self.processors[file_path] = processor
            processor.start()
    
    def cancel_processing(self, file_path):
        """取消特定影片的處理"""
        if file_path in self.processors:
            processor = self.processors[file_path]
            processor.cancel()
            
            # 更新UI
            widgets = self.progress_widgets[file_path]
            widgets['cancel_btn'].setEnabled(False)
            widgets['status_label'].setText("處理已取消")
            
            # 檢查是否所有處理都已完成
            self.check_all_completed()
    
    def check_all_completed(self):
        """檢查是否所有處理都已完成"""
        all_done = all(not p.isRunning() for p in self.processors.values())
        if all_done:
            self.process_btn.setEnabled(True)
            self.add_btn.setEnabled(True)
            self.remove_btn.setEnabled(True)
    
    def update_progress(self, file_path, value):
        """更新特定影片的進度"""
        if file_path in self.progress_widgets:
            self.progress_widgets[file_path]['progress_bar'].setValue(value)
    
    def processing_done(self, file_path, message):
        """處理完成後的操作"""
        if file_path in self.progress_widgets:
            widgets = self.progress_widgets[file_path]
            widgets['status_label'].setText(message)
            widgets['cancel_btn'].setEnabled(False)
        
        self.check_all_completed()

    def closeEvent(self, event):
        """關閉視窗時的處理"""
        # 取消所有正在進行的處理
        for processor in self.processors.values():
            if processor.isRunning():
                processor.cancel()
        
        # 等待所有處理器結束
        for processor in self.processors.values():
            processor.wait()
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 