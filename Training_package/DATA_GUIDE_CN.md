# Lychee-FD 训练数据格式

[English](DATA_GUIDE.md) · 中文

数据使用 HuggingFace `Dataset` 保存，每一行对应一段完整对话。

## 对话字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `user_audio_path` | 字符串 | 整段用户侧音轨的路径，推荐使用绝对路径 |
| `turns` | 列表 | 按时间排列的对话轮次，每轮包含 `user` 和 `assistant` |
| `turns[].user.start_time` | 数值 | 用户本轮语音的开始时间 |
| `turns[].user.end_time` | 数值 | 用户本轮语音的结束时间 |
| `turns[].assistant.start_time` | 数值 | AI 本轮回复的开始时间 |
| `turns[].assistant.text` | 字符串 | AI 回复文本 |
| `turns[].assistant.stoken` | 整数列表 | 与回复对应的 StepAudio 原始语音 token 编码 |
| `turns[].is_ai_been_interrupted` | 布尔值 | AI 回复是否被打断；为 `true` 时，下一轮用户的开始时间标记打断点 |
| `turns[].user_backchannel` | 列表 | 用户反馈语，每项包含 `start_time`、`end_time`、`clip_path` |
| `turns[].ai_backchannel` | 列表 | AI 反馈语，每项包含 `start_time`、`end_time`、`text`、`stoken` |

以上时间戳均以秒为单位，使用整段对话的全局时间基准。用户语音区间满足 `0 <= start_time < end_time <= 音轨时长`。

`clip_path` 指向独立的用户反馈语音片段。启用相应 backchannel 功能时提供对应列表，空列表写为 `[]`。

## 逐词时间戳

启用 `enable_force_align_pause` 时，提供以下两种格式之一。两者同时存在时，优先使用轮内字段。

| 字段 | 时间基准 |
| --- | --- |
| `turns[].user.force_align.timestamps` | 相对于本轮用户语音起点的局部秒数 |
| `user_audio_force_align.timestamps` | 相对于整段用户音轨起点的全局秒数 |

`timestamps` 为逐词 `[开始时间, 结束时间]` 列表。例如，本轮用户语音从第 10 秒开始：

```json
{
  "start_time": 10.0,
  "end_time": 13.0,
  "force_align": {
    "timestamps": [[0.2, 0.7], [1.5, 2.5]]
  }
}
```

对应的全局逐词时间为 `[[10.2, 10.7], [11.5, 12.5]]`。每个区间满足开始时间小于结束时间，空标注写为 `{"timestamps": []}`。

## 样例与保存

[conversation.json](examples/conversation.json) 展示完整字段结构。将示例中的音频路径、文本、时间戳和语音 token 替换为实际数据后保存：

```python
import json
from datasets import Dataset

with open("examples/conversation.json", encoding="utf-8") as f:
    records = json.load(f)

Dataset.from_list(records).save_to_disk("/path/to/processed_dataset")
```

训练时将 `DATA_PATH` 设置为保存目录；多个目录使用 `;` 分隔。
