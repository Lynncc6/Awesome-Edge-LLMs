# 🔍 Awesome Edge LLMs 
A comprehensive survey on Edge AI，covering hardware, software, frameworks, applications, performance optimization, and the deployment of LLMs on edge devices.


## Open Source Edge Models

The listed models are base model limited to either of the following:
- Parameter ≤ 10B
- Officially claimed edge models



| Model           | Size | Org | Time  | Download | Paper |
|:-----------:|:--:|:--:|:-----------:|:---------------:|:---------------:|
| Phi-4 | 14B | Microsoft | 2025.1.9 <br> 2024.12.12(release) | [🤗]https://huggingface.co/microsoft/phi-4 | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2412.08905) |
| [VITA-1.5](https://github.com/VITA-MLLM/VITA) | 7B | VITA | 2025.1.6 | - | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2501.01957) |
| [InternVL 2.5](https://github.com/OpenGVLab/InternVL) | 8B | OpenGVLab | 2024.12.22 | [🤗](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c) | - | 
| OmniAudio | 2.6B | Nexa AI | 2024.12.12 | [🤗](https://huggingface.co/NexaAIDev/OmniAudio-2.6B) | [📖](https://nexa.ai/blogs/omniaudio-2.6b) | 
| Ministral | 3B 8B | Mistral AI | 2024.10.16 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/ministraux/) |
| Pixtral 12B | 12B | Mistral AI | 2024.9.17 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/pixtral-12b/) |
| Phi 3.5 | 3.8B 4.1B | Microsoft | 2024.8.21 | [🤗](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | - | 
| [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V)  | 8B | OpenBMB | 2024.8.6 | [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6) | - |
| SmolLM | 135M 360M 1.7B | Hugging Face | 2024.8.2 | [🤗](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966) | [📖](https://huggingface.co/blog/smollm) |
| [Gemma2](https://github.com/google-deepmind/gemma) | 2B 9B | Google | 2024.7.31 | [🤗](https://huggingface.co/google/gemma-2-2b)|[📖](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)|
| [DCLM 7B](https://github.com/mlfoundations/dclm) | 7B | Apple | 2024.7.18 | [🤗](https://huggingface.co/apple/DCLM-7B) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.11794) |
| [Phi-3](https://github.com/microsoft/Phi-3CookBook/blob/main/README.md) | 3.8B 7B | Microsoft | 2024.4.23 | [🤗](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.14219)|
| Mistral NeMo | 12B | Mistral AI | 2024.6.18 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/mistral-nemo/) |
| [Gemma](https://github.com/google-deepmind/gemma) | 2B 7B | Google | 2024.2.21 |  [🤗](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)| [📖](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) |
| [Mistral 7B](https://github.com/mistralai/mistral-inference) | 2B 7B | Mistral AI | 2023.9.27 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/announcing-mistral-7b/) |



## LLM Inference 

|Title|Date|Org|Paper|
|:---:|:---:|:---:|:---:|
| [DashInfer-VLM](https://github.com/modelscope/dash-infer) | 2025.1 | ModelScope | [📖](https://dashinfer.readthedocs.io/en/latest/vlm/vlm_offline_inference_en.html) |
| SparseInfer | 2024.11 | University of Seoul, etc |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2411.12692) |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | 2024.6 | Moonshot AI | [📖](https://flashinfer.ai/2024/02/02/cascade-inference.html) |
| [flashinfer](https://github.com/flashinfer-ai/flashinfer) | 2024.2 | flashinfer-ai | [📖](https://flashinfer.ai/2024/02/02/cascade-inference.html)|
| [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | 2023.12 | SJTU | 
| [vLLM](https://github.com/vllm-project/vllm) | 2023.9 | UC Berkeley, etc | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2309.06180.pdf) | [📖](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf) |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | 2023.5 | mlc-ai | [📖](https://llm.mlc.ai/)  |
| [Ollama](https://github.com/ollama/ollama) | 2023.8 | Ollama Inc | - | 
| [LMDeploy](https://github.com/InternLM/lmdeploy) | 2023.6 | InternLM | [📖](https://lmdeploy.readthedocs.io/en/latest/) |





|Date|Title|Paper|Code|
|:---:|:---:|:---:|:---:|
| 2024.01 | [inferflow] INFERFLOW: AN EFFICIENT AND HIGHLY CONFIGURABLE INFERENCE ENGINE FOR LARGE LANGUAGE MODELS (@Tencent AI Lab) | [[pdf]](https://arxiv.org/pdf/2401.08294.pdf) | [![Stars](https://img.shields.io/github/stars/inferflow/inferflow.svg?style=social)](https://github.com/inferflow/inferflow) |
| 2023.12 | [**PETALS**] Distributed Inference and Fine-tuning of Large Language Models Over The Internet (@HSE University, etc) | [[pdf]](https://arxiv.org/pdf/2312.08361.pdf) | [![Stars](https://img.shields.io/github/stars/bigscience-workshop/petals.svg?style=social)](https://github.com/bigscience-workshop/petals) |
| 2023.10 | [**TensorRT-LLM**] NVIDIA TensorRT LLM (@NVIDIA) | [[docs]](https://nvidia.github.io/TensorRT-LLM/) | [![Stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social)](https://github.com/NVIDIA/TensorRT-LLM) |
| 2023.10 | [LightSeq] LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers (@UC Berkeley, etc) | [[pdf]](https://arxiv.org/pdf/2310.03294.pdf) | [![Stars](https://img.shields.io/github/stars/RulinShao/LightSeq.svg?style=social)](https://github.com/RulinShao/LightSeq) |
| 2023.09 | [StreamingLLM] EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS (@Meta AI, etc) | [[pdf]](https://arxiv.org/pdf/2309.17453.pdf) | [![Stars](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social)](https://github.com/mit-han-lab/streaming-llm) |
| 2023.09 | [Medusa] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads (@Tianle Cai, etc) | [[blog]](https://sites.google.com/view/medusa-llm) | [![Stars](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg?style=social)](https://github.com/FasterDecoding/Medusa) |
| 2023.08 | [**LightLLM**] LightLLM is a Python-based LLM (Large Language Model) inference and serving framework (@ModelTC) | [[docs]](https://github.com/ModelTC/lightllm) | [![Stars](https://img.shields.io/github/stars/ModelTC/lightllm.svg?style=social)](https://github.com/ModelTC/lightllm) |
| 2023.05 | [**SpecInfer**] Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification (@Peking University, etc) | [[pdf]](https://arxiv.org/pdf/2305.09781.pdf) | [![Stars](https://img.shields.io/github/stars/flexflow/FlexFlow.svg?style=social)](https://github.com/flexflow/FlexFlow/tree/inference) |
| 2023.05 | [**FastServe**] Fast Distributed Inference Serving for Large Language Models (@Peking University, etc) | [[pdf]](https://arxiv.org/pdf/2305.05920.pdf) | - |
| 2023.03 | [FlexGen] High-Throughput Generative Inference of Large Language Models with a Single GPU (@Stanford University, etc) | [[pdf]](https://arxiv.org/pdf/2303.06865.pdf) | [![Stars](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)](https://github.com/FMInference/FlexGen) |
| 2020.05 | [**Megatron-LM**] Training Multi-Billion Parameter Language Models Using Model Parallelism (@NVIDIA) | [[pdf]](https://arxiv.org/pdf/1909.08053.pdf) | [![Stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social)](https://github.com/NVIDIA/Megatron-LM) |


## Processor
### NVIDIA

✅ 50 Series @2025

|        | GeForce RTX 5090       | GeForce RTX 5080       | GeForce RTX 5070 Ti    | GeForce RTX 5070       |
|:-------------------------:|:------------------------:|:-------------------------:|:-------------------------:|:------------------------:|
| NVIDIA CUDA Cores       | 21760                 | 10752                  | 8960                   | 6144                  |
| Shader Cores            | Blackwell             | Blackwell              | Blackwell              | Blackwell             |
| Tensor Cores (AI)       | 5th Generation<br> 3352 AI TOPS         | 5th Generation <br> 1801 AI TOPS        | 5th Generation<br>1406 AI TOPS    | 5th Generation<br> 988 AI TOPS     |
| Ray Tracing Cores       | 4th Generation<br>318 TFLOPS          | 4th Generation<br>171 TFLOPS          | 4th Generation<br>133 TFLOPS          | 4th Generation<br>94 TFLOPS        |
| Boost Clock (GHz)       | 2.41                  | 2.62                   | 2.45                   | 2.51                  |
| Base Clock (GHz)        | 2.01                  | 2.30                   | 2.30                   | 2.16                  |
| Standard Memory Config  | 32 GB GDDR7           | 16 GB GDDR7            | 16 GB GDDR7            | 12 GB GDDR7           |
| Memory Interface Width  | 512-bit               | 256-bit                | 256-bit                | 192-bit               |
| Price                   | $1999                 | $999                   | $749                   | $549                   |

✅ 40 Super Series @2024


✅ 40 Series @2022

| GPU Specs               | GeForce RTX 4090       | GeForce RTX 4080      | GeForce RTX 4070 Ti    | GeForce RTX 4070      | GeForce RTX 4060 Ti   | GeForce RTX 4060 |  
|:-------------------------:|:------------------------:|:------------------------:|:------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
| NVIDIA CUDA Cores       | 16384                 | 9728                  | 7680                  | 5888                 | 4352                 | 3072                 |
| Shader Cores            | Ada Lovelace          | Ada Lovelace          | Ada Lovelace          | Ada Lovelace         | Ada Lovelace         | Ada Lovelace         |
| Tensor Cores (AI)       | 4th Gen<br>330 AI TFLOPS | 4th Gen<br>200 AI TFLOPS | 4th Gen<br>150 AI TFLOPS | 4th Gen<br>100 AI TFLOPS | 4th Gen<br>90 AI TFLOPS | 4th Gen<br>60 AI TFLOPS |
| Ray Tracing Cores       | 3rd Gen<br>191 TFLOPS | 3rd Gen<br>112 TFLOPS | 3rd Gen<br>92 TFLOPS  | 3rd Gen<br>64 TFLOPS | 3rd Gen<br>54 TFLOPS | 3rd Gen<br>35 TFLOPS |
| Boost Clock (GHz)       | 2.52                  | 2.51                  | 2.61                 | 2.48                 | 2.54                 | 2.42                 |
| Base Clock (GHz)        | 2.23                  | 2.21                  | 2.31                 | 1.92                 | 2.31                 | 1.83                 |
| Standard Memory Config  | 24 GB GDDR6X          | 16 GB GDDR6X          | 12 GB GDDR6X          | 12 GB GDDR6X         | 8 GB GDDR6           | 8 GB GDDR6           |
| Memory Interface Width  | 384-bit               | 256-bit               | 192-bit               | 192-bit              | 128-bit              | 128-bit              |
| Graphics Card Power (W) | 450W                  | 320W                  | 285W                  | 200W                 | 160W                 | 115W                 |
| Recommended PSU (W)     | 850W                  | 750W                  | 700W                  | 650W                 | 550W                 | 450W                 |
| Launch Price (USD)      | $1,599                | $1,199                | $799                  | $599                 | $399 (8GB)<br>$499 (16GB) | $299            |




## Hardware Applications

### AI Glasses
| Name  | Company | Model | Time  |  Price |
|:--:|:--:|:-----------:|:---------------:|:---------------:|
| [雷鸟V3](https://mp.weixin.qq.com/s/NBn81ocRqLhDKmtGXZuX0w) | 雷鸟创新 | Qwen | 2025.1.7 | ¥ 1799 + | 
| [闪极拍拍镜](https://mp.weixin.qq.com/s/exweWfZm1Eoc2LWM8tFsUg) | 闪极科技 | Qwen Kimi GLM, etc. | 2024.12.19 | ¥999 + |
| INMO GO2 | 影目科技 | - ｜ 2024.11.29 ｜ ¥3999 ｜
| Rokid Glasses | Rokid | Qwen | 2024.11.18 | ¥2499 |
| Looktech | Looktech | ChatGPT Claude Gemini | 2024.11.16 | $199 | 
| [Ray-Ban](https://www.ray-ban.com/usa) | Meta | Meta AI | 2023.9 | $299 | 














### References
[Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)

[数字生命卡兹克- AI硬件大全](https://datakhazix.feishu.cn/wiki/Zfp6wzb8eivwMqkSNgLcuiExnJd) @数字生命卡兹克










