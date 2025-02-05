import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from typing import List, Tuple
import logging
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import shutil
import subprocess
import sys

def check_system_requirements():
    """檢查系統需求和依賴項"""
    results = {
        'ffmpeg_python': False,
        'ffmpeg': False,
        'cuda': False,
        'python_version': sys.version,
        'opencv_version': cv2.__version__,
        'errors': []
    }
    
    # 檢查 ffmpeg-python
    try:
        import ffmpeg
        results['ffmpeg_python'] = True
        logging.info("✓ ffmpeg-python 已安裝")
    except ImportError as e:
        results['errors'].append(f"✗ ffmpeg-python 未安裝: {str(e)}")
    
    # 檢查系統 FFmpeg
    ffmpeg_paths = [
        'ffmpeg',  # 檢查 PATH
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',  # 檢查預設安裝路徑
        os.path.join(os.getenv('ProgramFiles'), 'ffmpeg', 'bin', 'ffmpeg.exe'),  # 使用環境變數
    ]
    
    ffmpeg_found = False
    for ffmpeg_path in ffmpeg_paths:
        try:
            result = subprocess.run([ffmpeg_path, '-version'], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                results['ffmpeg'] = True
                ffmpeg_found = True
                logging.info(f"✓ FFmpeg 已安裝: {version}")
                logging.info(f"  路徑: {ffmpeg_path}")
                break
        except FileNotFoundError:
            continue
        except Exception as e:
            logging.error(f"檢查 FFmpeg 時發生錯誤: {str(e)}")
            continue
    
    if not ffmpeg_found:
        results['errors'].append("""✗ 找不到 FFmpeg，請檢查:
1. 確認 FFmpeg 已安裝在 C:\\Program Files\\ffmpeg\\bin
2. 確認環境變數 PATH 包含 C:\\Program Files\\ffmpeg\\bin
3. 重新啟動電腦後再試""")
    
    # 檢查 CUDA 支援
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_count > 0:
        results['cuda'] = True
        logging.info(f"✓ 找到 {cuda_count} 個 CUDA 設備")
    else:
        logging.info("- 未檢測到 CUDA 設備，將使用 CPU 模式")
    
    # 輸出系統資訊
    logging.info(f"系統資訊:")
    logging.info(f"- Python 版本: {results['python_version']}")
    logging.info(f"- OpenCV 版本: {results['opencv_version']}")
    
    # 如果有錯誤，輸出錯誤信息
    if results['errors']:
        logging.error("\n需要解決的問題:")
        for error in results['errors']:
            logging.error(error)
    
    return results

# 在模組載入時進行系統檢查
SYSTEM_CHECK = check_system_requirements()
if SYSTEM_CHECK['errors']:
    logging.warning("系統檢查發現一些問題，請解決後再繼續")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logging.error("找不到 ffmpeg-python 套件，請執行 'pip install ffmpeg-python' 安裝")

class MotionDetector:
    def __init__(self):
        # 檢查CUDA是否可用
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            logging.info("使用 CUDA 加速動作檢測")
        
        # 光流法參數優化
        self.feature_params = dict(
            maxCorners=50,  # 減少特徵點數量
            qualityLevel=0.2,
            minDistance=10,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if self.use_cuda:
            self.stream = cv2.cuda.Stream()
            self.gpu_flow = cv2.cuda.SparsePyrLKOpticalFlow.create(
                winSize=self.lk_params['winSize'],
                maxLevel=self.lk_params['maxLevel']
            )
        
        self.prev_gray = None
        self.prev_points = None
    
    def calculate_motion_score(self, frame) -> float:
        """計算單幀影像中的動作幅度分數"""
        if self.use_cuda:
            return self._calculate_motion_score_gpu(frame)
        else:
            return self._calculate_motion_score_cpu(frame)
    
    def _calculate_motion_score_gpu(self, frame) -> float:
        """使用GPU計算動作分數"""
        try:
            # 轉換為灰度圖並上傳到GPU
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gpu_gray = cv2.cuda_GpuMat(gray)
            
            if self.prev_gray is None:
                self.prev_gray = gpu_gray
                gpu_points = cv2.cuda.GoodFeaturesToTrack_GPU(
                    gpu_gray,
                    maxCorners=self.feature_params['maxCorners'],
                    qualityLevel=self.feature_params['qualityLevel'],
                    minDistance=self.feature_params['minDistance'],
                    blockSize=self.feature_params['blockSize']
                )
                self.prev_points = gpu_points
                return 0.0
            
            if self.prev_points is None:
                gpu_points = cv2.cuda.GoodFeaturesToTrack_GPU(
                    self.prev_gray,
                    maxCorners=self.feature_params['maxCorners'],
                    qualityLevel=self.feature_params['qualityLevel'],
                    minDistance=self.feature_params['minDistance'],
                    blockSize=self.feature_params['blockSize']
                )
                self.prev_points = gpu_points
                if self.prev_points is None:
                    self.prev_gray = gpu_gray
                    return 0.0
            
            # 計算光流
            next_points, status = self.gpu_flow.calc(
                self.prev_gray, gpu_gray, self.prev_points, None, self.stream
            )
            
            if next_points is None:
                self.prev_gray = gpu_gray
                self.prev_points = None
                return 0.0
            
            # 下載結果到CPU進行計算
            next_points_cpu = next_points.download()
            prev_points_cpu = self.prev_points.download()
            status_cpu = status.download().ravel()
            
            # 選擇好的點
            good_new = next_points_cpu[status_cpu == 1]
            good_old = prev_points_cpu[status_cpu == 1]
            
            # 計算點的移動距離
            if len(good_new) > 0 and len(good_old) > 0:
                distances = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
                motion_score = np.mean(distances)
            else:
                motion_score = 0.0
            
            # 更新前一幀
            self.prev_gray = gpu_gray
            self.prev_points = next_points
            
            return motion_score
            
        except cv2.error as e:
            logging.error(f"GPU處理出錯，切換到CPU模式: {str(e)}")
            self.use_cuda = False
            return self._calculate_motion_score_cpu(frame)
    
    def _calculate_motion_score_cpu(self, frame) -> float:
        """使用CPU計算動作分數"""
        # 原始的CPU實現保持不變
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
        else:
            motion_score = 0.0
        
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return motion_score

class VideoAnalyzer:
    def __init__(self, input_path: str, clip_count: int = 5, min_duration: int = 10, max_duration: int = 60):
        """
        初始化影片分析器
        :param input_path: 輸入影片路徑
        :param clip_count: 要選取的片段數量
        :param min_duration: 最短片段時長（秒）
        :param max_duration: 最長片段時長（秒）
        """
        self.input_path = input_path
        self.clip_count = clip_count
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.motion_detector = MotionDetector()
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
        
        # 計算滑動窗口的大小（使用最小時長）
        window_size = int(fps * self.min_duration)
        
        # 儲存每一幀的動作分數
        frame_scores = []
        
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # 計算當前幀的動作分數
            motion_score = self.motion_detector.calculate_motion_score(frame)
            frame_scores.append(motion_score)
            
            frame_count += 1
            if self.progress_callback:
                progress = int((frame_count / total_frames) * 100)
                self.progress_callback(progress)
        
        video.release()
        
        # 使用滑動窗口計算平均分數
        window_scores = []
        for i in range(0, len(frame_scores) - window_size + 1):
            window = frame_scores[i:i + window_size]
            avg_score = np.mean(window)
            window_scores.append((i, avg_score))
        
        # 找出分數最高的連續片段
        time_segments = self._select_continuous_segments(window_scores, fps, total_frames)
        
        return time_segments
    
    def _select_continuous_segments(self, window_scores: List[Tuple[int, float]], fps: float, total_frames: int) -> List[Tuple[float, float]]:
        """選擇分數最高的連續片段"""
        if not window_scores:
            return []
        
        # 計算最小和最大片段的幀數
        min_frames = int(self.min_duration * fps)
        max_frames = int(self.max_duration * fps)
        
        # 將分數轉換為numpy數組以便計算
        scores = np.array([score for _, score in window_scores])
        
        # 使用滑動窗口找出平均分數最高的區間
        best_segments = []
        used_frames = set()
        
        for _ in range(self.clip_count):
            best_start = None
            best_score = -float('inf')
            best_length = min_frames
            
            # 嘗試不同的片段長度
            for length in range(min_frames, max_frames + 1, int(fps)):  # 每秒遞增
                # 計算每個可能的起始位置的平均分數
                for start in range(0, len(scores) - length + 1):
                    # 檢查是否與已選擇的片段重疊
                    if any(frame in used_frames for frame in range(start, start + length)):
                        continue
                    
                    avg_score = np.mean(scores[start:start + length])
                    if avg_score > best_score:
                        best_score = avg_score
                        best_start = start
                        best_length = length
            
            if best_start is not None:
                # 將這段幀標記為已使用
                used_frames.update(range(best_start, best_start + best_length))
                
                # 轉換為秒
                start_time = best_start / fps
                end_time = (best_start + best_length) / fps
                
                best_segments.append((start_time, end_time))
            else:
                break
        
        # 按時間順序排序片段
        best_segments.sort(key=lambda x: x[0])
        
        return best_segments

def process_video_segment(args):
    """單個影片片段的處理函數（用於多進程）"""
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("找不到 ffmpeg-python 套件")
    
    if not SYSTEM_CHECK['ffmpeg']:
        raise RuntimeError("系統中找不到 FFmpeg")
    
    input_path, output_path, start_time, end_time = args
    try:
        # 使用 ffmpeg-python 直接處理，避免中間檔案
        stream = ffmpeg.input(input_path, ss=start_time, t=end_time-start_time)
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec='h264_nvenc' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'libx264',
            acodec='aac',
            video_bitrate='5M',
            audio_bitrate='160k',
            preset='fast' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'ultrafast',
            threads=multiprocessing.cpu_count(),
            loglevel='error'
        )
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        return True
    except Exception as e:
        logging.error(f"處理片段時發生錯誤: {str(e)}")
        return False

def process_video(input_path: str, output_path: str, progress_callback=None, 
                 clip_count: int = 5, min_duration: int = 10, max_duration: int = 60,
                 is_cancelled_callback=None) -> bool:
    """
    處理影片的主函數
    :param input_path: 輸入影片路徑
    :param output_path: 輸出資料夾路徑
    :param progress_callback: 進度回調函數
    :param clip_count: 要選取的片段數量
    :param min_duration: 最短片段時長（秒）
    :param max_duration: 最長片段時長（秒）
    :param is_cancelled_callback: 檢查是否已取消的回調函數
    :return: 是否成功
    """
    try:
        # 檢查必要的依賴
        if not FFMPEG_AVAILABLE:
            raise RuntimeError("找不到 ffmpeg-python 套件，請執行 'pip install ffmpeg-python' 安裝")
        
        if not SYSTEM_CHECK['ffmpeg']:
            raise RuntimeError("系統中找不到 FFmpeg，請參考錯誤訊息進行安裝")
        
        # 創建輸出資料夾
        output_dir = output_path.replace(".mp4", "_clips")
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分析影片
        analyzer = VideoAnalyzer(input_path, clip_count, min_duration, max_duration)
        if progress_callback:
            analyzer.set_progress_callback(progress_callback)
        
        segments = analyzer.analyze_video()
        
        # 檢查是否已取消
        if is_cancelled_callback and is_cancelled_callback():
            return False
        
        if not segments:
            logging.error("沒有找到有效的片段")
            return False
        
        # 準備多進程處理的參數
        process_args = []
        for i, (start_time, end_time) in enumerate(segments):
            output_filename = f"clip_{i+1:02d}_{start_time:.1f}-{end_time:.1f}.mp4"
            output_file_path = os.path.join(output_dir, output_filename)
            process_args.append((input_path, output_file_path, start_time, end_time))
        
        # 使用進程池處理片段
        with ProcessPoolExecutor(max_workers=min(len(segments), multiprocessing.cpu_count())) as executor:
            # 提交所有任務
            future_to_args = {
                executor.submit(process_video_segment, args): args 
                for args in process_args
            }
            
            # 等待結果
            results = []
            for future in future_to_args:
                try:
                    # 檢查是否已取消
                    if is_cancelled_callback and is_cancelled_callback():
                        executor.shutdown(wait=False)
                        return False
                    
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"處理片段時發生錯誤: {str(e)}")
                    results.append(False)
        
        # 檢查處理結果
        if not any(results):
            raise Exception("所有片段處理都失敗了")
        
        return True
        
    except Exception as e:
        logging.error(f"處理影片時發生錯誤: {str(e)}")
        return False 