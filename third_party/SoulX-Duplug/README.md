<div align="center">
    <h1>
    SoulX-Duplug
    </h1>
    <p>
    Official code for enabling full-duplex speech interaction with<br>
    <b><em>SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation</em></b>
    </p>
    <p>
    <img src="assets/SoulX-Duplug-logo.png" alt="SoulX-Duplug Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <!-- <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/Python-3.10-blue" alt="version"></a> -->
    <a href="https://soulx-duplug.sjtuxlance.com/"><img src="https://img.shields.io/badge/🌐%20Online-Demo-blue" alt="Online Demo"></a>
    <a href="https://arxiv.org/abs/2603.14877"><img src="https://img.shields.io/badge/arXiv-2603.14877-B31B1B?logo=arxiv&logoColor=white.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="HF-Model"></a>
    <a href="https://huggingface.co/datasets/Soul-AILab/SoulX-Duplug-Eval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Eval-yellow" alt="HF-Eval"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache-2.0"></a>
</div>


## ✨ Overview
SoulX-Duplug is a **plug-and-play streaming semantic VAD model** designed for real-time full-duplex speech conversation. Through text-guided streaming state prediction, SoulX-Duplug enables low-latency, semantic-aware streaming dialogue management. In addition to the core model, we also **open-source a [dialogue system](https://github.com/Soul-AILab/SoulX-Duplug/tree/dialogue-system) build on top of SoulX-Duplug**, which demonstrates the practicality of our model in real-world applications.

To facilitate benchmarking and research in this area, we also release **[SoulX-Duplug-Eval](https://huggingface.co/datasets/Soul-AILab/SoulX-Duplug-Eval)**, a complementary evaluation set for benchmarking full-duplex spoken dialogue systems.



## 🔥 Demo

Below is a demo of **full-duplex speech interaction** powered by SoulX-Duplug.

https://github.com/user-attachments/assets/cf5b040e-bc87-4fa9-ae6f-669db80a49eb

You can also try the **online interactive demo** here:

👉 https://soulx-duplug.sjtuxlance.com/


## 🚀 News
- **[2026-03-17]** Our paper on this project has been published! You can read it here: [SoulX-Duplug](https://arxiv.org/abs/2603.14877).
- **[2026-03-16]** SoulX-Duplug checkpoint and SoulX-Duplug-Eval are now available on Hugging Face! You can access it directly from [SoulX-Duplug-HF](https://huggingface.co/collections/Soul-AILab/soulx-duplug).


## 🛠️ Install

### Clone and Install
Here are instructions for installing on Linux.

- Clone the repo
```bash
git clone https://github.com/Soul-AILab/SoulX-Duplug.git
cd SoulX-Duplug
```

- Install system dependencies
```bash
sudo apt-get update
sudo apt-get install ffmpeg sox libsox-dev -y
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html

- Create Conda env
```bash
conda create -n soulx-duplug -y python=3.10
conda activate soulx-duplug
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```



### Model Download

Download via hf:
```bash
# If you are in mainland China, please first set the mirror:
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Soul-AILab/SoulX-Duplug-0.6B --local-dir pretrained_models
```

Download via python:
```python
from huggingface_hub import snapshot_download
snapshot_download("Soul-AILab/SoulX-Duplug-0.6B", local_dir="pretrained_models") 
```

Download via git clone:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B pretrained_models
```


### Configuration Details

In `config/config.yaml`:

- For the `infer_config.asr` field:
    - For Chinese, we recommend using `model_name: paraformer`
    - For English, set it to `model_name: sensevoice`, `language: en`
    - For bilingual scenarios, use `model_name: sensevoice`, `language: auto`

- The `max_wait_num` parameter is used as a fallback mechanism to handle potential misclassification of *incomplete* cases. It defines the number of chunks to wait without additional user speech before the assistant starts responding.

- The `far_field_threshold` parameter sets the threshold for filtering far-field audio in noisy environments.


### Basic Usage
We provide a streaming inference server for SoulX-Duplug. Start the server:
```bash
bash run.sh
```

For usage (see [example_client.py](https://github.com/Soul-AILab/SoulX-Duplug/blob/main/example_client.py) for reference), streamingly send your audio query (in chunks) to the server, and the server will return its prediction of the current dialogue state in a `dict`:

- Format:
    ```python
    {
        "type": "turn_state",
        "session_id": ,         # session_id
        "state": {
            "state": ,          # predicted state: "idle", "nonidle", "speak", or "blank"
            "text": ,           # (optional) asr result of user's turn
            "asr_segment": ,    # (optional) asr result of current chunk
            "asr_buffer": ,     # (optional) asr result of last 3.2s
        },
        "ts": time.time(),      # timestamp
    }
    ```

- **"idle"** indicates that the current audio chunk contains no semantic content (e.g., silence, noise, or backchannel).

- **"nonidle"** indicates that the current audio chunk contains semantic content. In this case, `"asr_segment"` returns the ASR result of the current chunk, and `"asr_buffer"` returns the ASR result of the accumulated audio over the past 3.2 seconds.

- **"speak"** indicates that up to the current chunk, the user is judged to have stopped speaking and the utterance is semantically complete, meaning the system can take the turn. In this case, `"asr_segment"` returns the ASR result of the current chunk, `"asr_buffer"` returns the ASR result of the accumulated audio over the past 3.2 seconds, and `"text"` returns the complete transcription of the user’s utterance for this turn. 

- **"blank"** indicates that the current unprocessed streaming input does not yet fill a full chunk; the server has cached the input and is waiting for the next query.


### Dialogue System

We implemented a demo full-duplex spoken dialogue system based on SoulX-Duplug. See the [`dialogue-system` branch](https://github.com/Soul-AILab/SoulX-Duplug/blob/dialogue-system/dialogue_system/README.md) for the demo code.


## 📌 TODOs
- [x] Publish the technical report.
- [x] Release evaluation scripts.


## 🔖 Citation
If you find this work useful in your research, please consider citing:

```bibtex
@misc{yan2026soulxduplug,
      title={SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation}, 
      author={Ruiqi Yan and Wenxi Chen and Zhanxun Liu and Ziyang Ma and Haopeng Lin and Hanlin Wen and Hanke Xie and Jun Wu and Yuzhe Liang and Yuxiang Zhao and Pengchao Feng and Jiale Qian and Hao Meng and Yuhang Dai and Shunshun Yin and Ming Tao and Lei Xie and Kai Yu and Xinsheng Wang and Xie Chen},
      year={2026},
      eprint={2603.14877},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.14877}, 
}
```

## 📜 License
This project is licensed under the [Apache 2.0 License](LICENSE).


## 🙏 Acknowledgment
We thank the following open-source projects for their open-source contributions:

- [QwenLM](https://github.com/QwenLM)
- [GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice)
- [chinese_text_normalization](https://github.com/speechio/chinese_text_normalization)
- [Paraformer](https://github.com/modelscope/FunASR/wiki/paraformer)
- [Sensevoice](https://github.com/FunAudioLLM/SenseVoice)
- [ChatTTS](https://github.com/2noise/ChatTTS)
- [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM)
