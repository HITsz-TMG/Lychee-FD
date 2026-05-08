import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import datasets
import os
import warnings


def inject_ambient_noise(
    audio: np.ndarray,
    sr: int = 16000,
    target_dbfs: float = -23.0,      # 归一化目标音量(EBU R128标准)
    snr_db_range: tuple = (40, 60),  # 底噪SNR范围
    noise_mix: dict = None,          # 噪声混合比例
    normalize: bool = True,
    speech_segments: list = None,    # 说话时间段列表 [(start_sec, end_sec), ...]
    eps: float = 1e-7,
) -> np.ndarray:
    """
    对音频注入模拟真实环境的微弱底噪
    
    Args:
        audio: 输入音频 [-1, 1] 范围的 float32
        sr: 采样率
        target_dbfs: 归一化目标响度(dBFS),-23是广播标准
        snr_db_range: 底噪SNR范围,40-60dB几乎听不出
        noise_mix: 噪声混合比例,默认 {'white': 0.3, 'pink': 0.7}
        normalize: 是否先做响度归一化
    
    Returns:
        加噪后的音频
    """
    if noise_mix is None:
        noise_mix = {'white': 0.3, 'pink': 0.7}
    
    audio = audio.astype(np.float32)

    assert audio.ndim == 1, f"仅支持单声道音频，当前 shape={audio.shape}"
    
    # ============ Step 1: 响度归一化 ============
    if normalize:
        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(audio)
        assert current_lufs > -70  # 有效音频
        audio = pyln.normalize.loudness(audio, current_lufs, target_dbfs)

    # ============ Step 2: 生成混合噪声 ============
    length = len(audio)
    noise = np.zeros(length, dtype=np.float32)
    
    if noise_mix.get('white', 0) > 0:
        white = np.random.randn(length).astype(np.float32)
        white = white / (np.std(white) + eps)  # 归一化到单位方差
        noise += noise_mix['white'] * white
    
    if noise_mix.get('pink', 0) > 0:
        pink = generate_pink_noise(length).astype(np.float32)
        pink = pink / (np.std(pink) + eps)
        noise += noise_mix['pink'] * pink
    
    # 噪声整体归一化到单位方差
    noise = noise / (np.std(noise) + eps)
    
    # ============ Step 3: 按SNR缩放噪声 ============
    snr_db = np.random.uniform(*snr_db_range)
    # 用提供的说话时间段精确提取有效语音采样点
    active_samples = []
    for start_sec, end_sec in speech_segments:
        s = int(start_sec * sr)
        e = int(end_sec * sr)
        s = max(0, min(s, length))
        e = max(s, min(e, length))
        if e > s:
            active_samples.append(audio[s:e])
            
    assert len(active_samples) > 0
    active_audio = np.concatenate(active_samples)
    signal_power = np.mean(active_audio ** 2) + eps
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power)
    
    # ============ Step 4: 混合 & 防削波 ============
    output = audio + noise
    
    peak = np.abs(output).max()
    if peak > 0.99:
        output = output * (0.99 / peak)
    
    return output


def generate_pink_noise(length: int) -> np.ndarray:
    """
    用 Voss-McCartney 算法生成粉红噪声(快速且不需要额外依赖)
    也可以用 FFT 方法,这里用FFT滤波法更标准
    """
    # FFT方法:白噪声经过 1/sqrt(f) 滤波
    white = np.random.randn(length)
    
    # 转到频域
    f = np.fft.rfftfreq(length)
    f[0] = f[1]  # 避免除零
    
    # 1/f 功率谱 → 1/sqrt(f) 幅度谱
    spectrum = np.fft.rfft(white)
    spectrum = spectrum / np.sqrt(f)
    
    pink = np.fft.irfft(spectrum, n=length)
    return pink


def parse_timestamp(ts_str):
    """
    解析时间戳字符串, 支持两种格式:
    "HH:MM:SS,mmm" (逗号分隔毫秒) 或 "HH:MM:SS.mmm" (点分隔毫秒)
    返回秒数(float)
    """
    ts_str = ts_str.strip()
    ts_str = ts_str.replace(',', '.')
    parts = ts_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

