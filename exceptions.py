class VideoProcessingException(Exception):
    """影片處理異常基類"""
    pass

class InvalidVideoFormatError(VideoProcessingException):
    """無效影片格式異常"""
    def __init__(self, file_path):
        super().__init__(f"不支援的影片格式: {file_path}")

class InsufficientMotionError(VideoProcessingException):
    """動作不足異常"""
    def __init__(self):
        super().__init__("未檢測到足夠動作片段") 