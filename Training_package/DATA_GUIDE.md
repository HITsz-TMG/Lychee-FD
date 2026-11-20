# Lychee-FD Training Data Format

English · [中文](DATA_GUIDE_CN.md)

Store data as a HuggingFace `Dataset`, with one complete conversation per row.

## Conversation fields

| Field | Type | Description |
| --- | --- | --- |
| `user_audio_path` | String | Path to the full user-side audio track; an absolute path is recommended |
| `turns` | List | Conversation turns in chronological order, each containing `user` and `assistant` |
| `turns[].user.start_time` | Number | Start time of the user's speech in this turn |
| `turns[].user.end_time` | Number | End time of the user's speech in this turn |
| `turns[].assistant.start_time` | Number | Start time of the AI response in this turn |
| `turns[].assistant.text` | String | AI response text |
| `turns[].assistant.stoken` | List of integers | Raw StepAudio speech-token codes corresponding to the response |
| `turns[].is_ai_been_interrupted` | Boolean | Whether the AI response is interrupted; when `true`, the next user turn's start time marks the interruption |
| `turns[].user_backchannel` | List | User backchannels; each entry contains `start_time`, `end_time`, and `clip_path` |
| `turns[].ai_backchannel` | List | AI backchannels; each entry contains `start_time`, `end_time`, `text`, and `stoken` |

All timestamps above are in seconds on the full conversation timeline. User speech intervals satisfy `0 <= start_time < end_time <= audio duration`.

`clip_path` points to a separate user backchannel audio clip. Supply the corresponding list when enabling a backchannel feature; represent an empty list as `[]`.

## Word timestamps

When `enable_force_align_pause` is enabled, supply either of the following formats. The turn-level field takes precedence when both are present.

| Field | Time reference |
| --- | --- |
| `turns[].user.force_align.timestamps` | Local seconds relative to the start of the current user speech segment |
| `user_audio_force_align.timestamps` | Global seconds relative to the start of the full user audio track |

`timestamps` is a list of word-level `[start, end]` pairs. For example, for a user turn starting at 10 seconds:

```json
{
  "start_time": 10.0,
  "end_time": 13.0,
  "force_align": {
    "timestamps": [[0.2, 0.7], [1.5, 2.5]]
  }
}
```

The corresponding global word timestamps are `[[10.2, 10.7], [11.5, 12.5]]`. Each interval has a start time earlier than its end time. Represent an empty annotation as `{"timestamps": []}`.

## Example and export

[conversation.json](examples/conversation.json) shows the complete field structure. Replace the example audio paths, text, timestamps, and speech tokens with your data, then save:

```python
import json
from datasets import Dataset

with open("examples/conversation.json", encoding="utf-8") as f:
    records = json.load(f)

Dataset.from_list(records).save_to_disk("/path/to/processed_dataset")
```

Set `DATA_PATH` to the saved directory for training. Separate multiple directories with `;`.
