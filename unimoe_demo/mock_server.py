import asyncio
import websockets
import json
import time
import os

# ====== 设置并创建保存 JSON 文件的目录 ======
output_dir = "JSON_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# ==================================================

async def handle_client(websocket):
    print("🌟 前端已连接，开始 JSON 协议测试！")
    
    # ====== 新增：用于缓存数据的列表和计时器 ======
    json_buffer = []
    last_save_time = time.time()
    # ==============================================
    
    try:
        async for message in websocket:
            # 1. 解析前端发来的上行 JSON 包
            try:
                data = json.loads(message)
                seq_id = data.get("seq_id", 0)
                session_id = data.get("session_id", "unknown")
                audio_payload = data.get("audio", {}).get("payload", "")
                
                print(f"📥 收到上行包: Session={session_id[:8]}..., Seq={seq_id}, 音频长度={len(audio_payload)}")

                # ====== 新增：将数据加入缓存，每 3 秒保存一次 ======
                json_buffer.append(data)
                
                current_time = time.time()
                if current_time - last_save_time >= 3.0:
                    timestamp = int(current_time * 1000)
                    # 文件名标明这是一个批次文件
                    filename = f"req_batch_{session_id[:8]}_{timestamp}.json"
                    filepath = os.path.join(output_dir, filename)
                    
                    # 将过去 3 秒收集到的所有包存为一个 JSON 数组
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(json_buffer, f, ensure_ascii=False, indent=2)
                    
                    print(f"💾 已保存过去的 3 秒数据，共 {len(json_buffer)} 个包到 {filename}")
                    
                    # 清空缓存池，重置计时器
                    json_buffer = []
                    last_save_time = current_time
                # ====================================================

                # 2. 组装符合后端规范的下行 JSON 包
                response_payload = {
                    "seq_id": seq_id,
                    "timestamp": int(time.time() * 1000),
                    "text": f"测试回复{seq_id}。 ",  # 模拟 AI 打字机文字
                    "audio": {
                        "format": "pcm_16k_16bit",
                        "payload": audio_payload    # 把收到的音频直接弹回去当回音
                    },
                    "meta_data": {
                        "status": "speaking"
                    }
                }

                # 3. 发回给前端
                await websocket.send(json.dumps(response_payload))

            except json.JSONDecodeError:
                print("❌ 错误：收到的不是合法的 JSON 格式！")
                
    except websockets.exceptions.ConnectionClosed:
        print("⭕ 前端连接已断开")
        
        # ====== 新增：断开连接时，如果缓存里还有没存完的数据，做一次收尾保存 ======
        if len(json_buffer) > 0:
            timestamp = int(time.time() * 1000)
            filename = f"req_batch_{session_id[:8]}_{timestamp}_final.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_buffer, f, ensure_ascii=False, indent=2)
            print(f"💾 通话结束，已保存剩余的 {len(json_buffer)} 个包到 {filename}")
        # ========================================================================

async def main():
    # 监听 8083 端口
    async with websockets.serve(handle_client, "localhost", 8083):
        print(f"🚀 Mock Server 已启动，监听 ws://localhost:8083 ...")
        print(f"📁 收到的 JSON 数据将【每 3 秒打包一次】保存在 '{output_dir}' 文件夹下")
        await asyncio.Future()  # 永久运行

if __name__ == "__main__":
    asyncio.run(main())