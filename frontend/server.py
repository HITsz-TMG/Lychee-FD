import asyncio
import websockets
import numpy as np
import wave
import io
import os
import time

# 设置保存音频文件的目录
output_dir = "audio_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 音频数据缓存
audio_data_buffer = []
audio_data_duration = 0  # 已接收的音频数据时长（秒）

async def handle_client(websocket):
    global audio_data_buffer, audio_data_duration
    print("用户已连接")

    try:
        while True:
            # 接收前端传来的音频数据（每160ms一次）
            audio_data = await websocket.recv()

            # 将接收到的音频数据转换为 Float32 数组
            audio_data_float32 = np.frombuffer(audio_data, dtype=np.float32)

            # 将音频数据添加到缓存中
            audio_data_buffer.extend(audio_data_float32)
            print("接收到音频数据")
            audio_data_duration += 0.16  # 假设每个数据块代表 160ms，所以每次接收增加0.16秒

            # 每5秒保存一次音频数据
            if audio_data_duration >= 5:
                # 1. 保存文件并获取标准 WAV 数据
                wav_bytes = save_audio_to_file(audio_data_buffer) 
                
                # 2. 将这完整的 5秒 WAV 文件发回给前端进行播放！
                await websocket.send(wav_bytes) 
                print("已向前端发送5秒钟的完整WAV音频进行播放！")
                
                audio_data_buffer = []  
                audio_data_duration = 0

    except websockets.exceptions.ConnectionClosed as e:
        print(f"连接关闭，状态码: {e.code}, 原因: {e.reason}")

def save_audio_to_file(audio_data):
    """将音频数据保存为WAV文件"""
    # 将接收到的音频数据从 Float32 转为 int16
    audio_data_int16 = np.int16(np.array(audio_data) * 32767)  # Convert float32 to int16

    # 创建一个内存中的音频文件
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16 位样本
        wf.setframerate(44100)  # 设置采样率为 44.1kHz
        wf.writeframes(audio_data_int16.tobytes())

    # 生成文件名，使用时间戳
    timestamp = int(time.time())
    output_file_path = os.path.join(output_dir, f"audio_{timestamp}.wav")

    # 获取标准的WAV格式字节流
    wav_bytes = byte_io.getvalue() 

    # 将音频数据写入文件
    with open(output_file_path, "wb") as f:
        f.write(wav_bytes)

    print(f"音频数据已保存为 {output_file_path}")

    # 返回完整的WAV文件数据
    return wav_bytes 

async def main():
    stop_event = asyncio.Event()
    async with websockets.serve(handle_client, "localhost", 8083):
        print("服务器启动，等待客户端连接...")
        await stop_event.wait()  # 等待关闭信号

if __name__ == "__main__":
    asyncio.run(main())