import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from typing import List, Tuple
import logging
import os
from dataclasses import dataclass

@dataclass
class ProcessingParams:
    """影片處理參數"""
    clip_duration: int = 60  # 片段長度（秒）
    num_clips: int = 5      # 選取片段數量
    motion_threshold: float = 0.3  # 動作檢測閾值
    quality_preset: str = 'veryfast'  # 編碼預設值

class MotionDetector:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        # 光流法參數
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_points = None
    
    def calculate_motion_score(self, frame) -> float:
        """計算單幀影像中的動作幅度分數"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return 0.0
        
        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.prev_points is None:
                self.prev_gray = gray
                return 0.0
        
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if next_points is None:
            self.prev_gray = gray
            self.prev_points = None
            return 0.0
        
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        if len(good_new) > 0 and len(good_old) > 0:
            distances = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
            motion_score = np.mean(distances)
            # 應用閾值
            motion_score = motion_score if motion_score > self.threshold else 0.0
        else:
            motion_score = 0.0
        
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return motion_score

class VideoAnalyzer:
    def __init__(self, input_path: str, params: ProcessingParams):
        """
        初始化影片分析器
        :param input_path: 輸入影片路徑
        :param params: 處理參數
        """
        self.input_path = input_path
        self.params = params
        self.motion_detector = MotionDetector(threshold=params.motion_threshold)
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """設置進度回調函數"""
        self.progress_callback = callback
    
    def analyze_video(self) -> List[Tuple[float, float]]:
        """
        分析影片並返回動作分數較高的時間段
        :return: 列表of元組 (開始時間, 結束時間)
        """
        video = cv2.VideoCapture(self.input_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 計算每個片段的幀數
        frames_per_segment = int(fps * self.params.clip_duration)
        
        # 儲存每個片段的平均動作分數
        segment_scores = []
        current_segment_frames = []
        
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # 計算當前幀的動作分數
            motion_score = self.motion_detector.calculate_motion_score(frame)
            current_segment_frames.append(motion_score)
            
            # 如果累積了足夠的幀數，計算這個片段的平均分數
            if len(current_segment_frames) >= frames_per_segment:
                avg_score = np.mean(current_segment_frames)
                segment_scores.append(avg_score)
                current_segment_frames = []
            
            frame_count += 1
            if self.progress_callback:
                progress = int((frame_count / total_frames) * 50)  # 分析階段佔總進度的50%
                self.progress_callback(progress)
        
        video.release()
        
        # 選擇動作分數最高的幾個片段
        top_segments = self._select_top_segments(segment_scores, self.params.num_clips)
        
        # 轉換為時間戳
        time_segments = []
        for segment_idx in top_segments:
            start_time = segment_idx * self.params.clip_duration
            end_time = start_time + self.params.clip_duration
            time_segments.append((start_time, end_time))
        
        return time_segments
    
    def _select_top_segments(self, scores: List[float], n: int) -> List[int]:
        """選擇分數最高的n個片段索引"""
        # 將分數和索引配對
        indexed_scores = list(enumerate(scores))
        # 按分數排序
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        # 取前n個索引
        return [idx for idx, _ in indexed_scores[:n]]

def process_video(input_path: str, output_path: str, progress_callback=None, params: ProcessingParams = None) -> bool:
    """
    處理影片的主函數
    :param input_path: 輸入影片路徑
    :param output_path: 輸出資料夾路徑
    :param progress_callback: 進度回調函數
    :param params: 處理參數
    :return: 是否成功
    """
    try:
        if params is None:
            params = ProcessingParams()
        
        # 創建輸出資料夾
        output_dir = output_path.replace(".mp4", "_clips")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分析影片
        analyzer = VideoAnalyzer(input_path, params)
        if progress_callback:
            analyzer.set_progress_callback(progress_callback)
        
        segments = analyzer.analyze_video()
        
        # 使用 CPU 編碼，但優化參數
        codec = 'libx264'
        preset = params.quality_preset
        extra_args = [
            '-crf', '23',  # 控制視頻質量
            '-tune', 'fastdecode',  # 優化解碼速度
            '-threads', '0'  # 使用所有可用的 CPU 核心
        ]
        
        logging.info(f"使用編碼器: {codec}, 預設值: {preset}")
        
        # 使用MoviePy剪輯影片
        video = VideoFileClip(input_path)
        total_duration = video.duration
        logging.info(f"影片總長度: {total_duration:.1f}秒")
        
        # 處理每個片段
        for i, (start_time, end_time) in enumerate(segments, 1):
            try:
                # 確保時間範圍有效
                if start_time >= total_duration or end_time > total_duration:
                    logging.warning(f"片段 {i} 時間範圍無效: {start_time:.1f}s - {end_time:.1f}s")
                    continue
                
                output_file = os.path.join(output_dir, f"clip_{i:02d}.mp4")
                logging.info(f"處理片段 {i}: {start_time:.1f}s - {end_time:.1f}s")
                
                # 剪輯片段
                clip = video.subclip(start_time, end_time)
                
                # 輸出片段
                clip.write_videofile(
                    output_file,
                    codec=codec,
                    preset=preset,
                    audio_codec='aac',
                    ffmpeg_params=extra_args,
                    verbose=False,
                    logger=None,
                    threads=4  # 使用多線程加速
                )
                
                clip.close()
                logging.info(f"片段 {i} 處理完成")
                
                if progress_callback:
                    # 更新總進度（分析佔50%，剪輯佔50%）
                    progress = 50 + int((i / len(segments)) * 50)
                    progress_callback(progress)
                    
            except Exception as clip_error:
                logging.error(f"處理片段 {i} 時發生錯誤: {str(clip_error)}")
                continue
        
        # 清理資源
        video.close()
        
        # 檢查輸出結果
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        if not output_files:
            raise Exception("沒有成功輸出任何片段")
        else:
            logging.info(f"成功輸出 {len(output_files)} 個片段")
        
        return True
        
    except Exception as e:
        logging.error(f"處理影片時發生錯誤: {str(e)}")
        return False 