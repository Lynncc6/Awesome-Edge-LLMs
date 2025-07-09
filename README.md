# 🔍 Awesome Edge LLMs 
A comprehensive survey on Edge AI，covering hardware, software, frameworks, applications, performance optimization, and the deployment of LLMs on edge devices.


## Open Source Edge Models

The listed models are base model limited to either of the following:
- Parameter ≤ 10B
- Officially claimed edge models



| Model           | Size | Org | Time  | Download | Paper |
|:-----------:|:--:|:--:|:-----------:|:---------------:|:---------------:|
| [MiniCPM4](https://github.com/OpenBMB/MiniCPM) | 8B | OpenBMB | 2025.6.6 | [🤗]([https://huggingface.co/openbmb/MiniCPM4](https://huggingface.co/collections/openbmb/minicpm4-6841ab29d180257e940baa9b)) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.07900) |
| [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | 7B | Qwen | 2025.3.26 | [🤗](https://huggingface.co/collections/Qwen/qwen25-omni-67de1e5f0f9464dc6314b36e) |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.20215)  |
| [MiniCPM-o 2.6](https://github.com/OpenBMB/MiniCPM-o) | 8B | OpenBMB | 2025.1.14 | [🤗](https://huggingface.co/openbmb/MiniCPM-o-2_6) | - |
| Phi-4 | 14B | Microsoft | 2025.1.9 <br> 2024.12.12(release) | [🤗](https://huggingface.co/microsoft/phi-4) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2412.08905) |
| [VITA-1.5](https://github.com/VITA-MLLM/VITA) | 7B | VITA | 2025.1.6 | - | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2501.01957) |
| [Megrez-3B-Omni](https://github.com/infinigence/Infini-Megrez) | 3B | Infinigence | 2024.12.16 | [🤗](https://huggingface.co/Infinigence/Megrez-3B-Omni) | - |
| OmniAudio | 2.6B | Nexa AI | 2024.12.12 | [🤗](https://huggingface.co/NexaAIDev/OmniAudio-2.6B) | [📖](https://nexa.ai/blogs/omniaudio-2.6b) | 
| [InternVL 2.5](https://github.com/OpenGVLab/InternVL) | 8B | OpenGVLab | 2024.12.5 | [🤗](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c) | - | 
| [GLM-Edge](https://github.com/THUDM/GLM-Edge?tab=readme-ov-file) | 1.5B 2B 4B 5B | THUDM | 2024.11.29 | [🤗](https://huggingface.co/THUDM/glm-edge-1.5b-chat) | - | 
| SmalVLM | 2B | Hugging Face | 2024.11.26 | [🤗](https://huggingface.co/HuggingFaceTB/SmolVLM-Base) | [📖](https://huggingface.co/blog/smolvlm) |
| [SmalLM2](https://github.com/huggingface/smollm) | 135M 360M 1.7B | Hugging Face | 2024.11.1 | [🤗](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966) | [📖](https://huggingface.co/blog/smollm) |
| Ministral | 3B 8B | Mistral AI | 2024.10.16 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/ministraux/) |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B | Qwen | 2024.9.19 | [🤗](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) | [📖](https://qwenlm.github.io/blog/qwen2.5/)|
| Pixtral 12B | 12B | Mistral AI | 2024.9.17 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/pixtral-12b/) |
| Qwen2-VL | 2B 7B | Qwen | 2024.8.30 | [🤗](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d) | [📖](https://qwenlm.github.io/blog/qwen2-vl/)
| Phi 3.5 | 3.8B 4.1B | Microsoft | 2024.8.21 | [🤗](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | - | 
| [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V)  | 8B | OpenBMB | 2024.8.6 | [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6) | - |
| [SmolLM](https://github.com/huggingface/smollm) | 135M 360M 1.7B | Hugging Face | 2024.8.2 | [🤗](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966) | [📖](https://huggingface.co/blog/smollm) |
| [Gemma2](https://github.com/google-deepmind/gemma) | 2B 9B | Google | 2024.7.31 | [🤗](https://huggingface.co/google/gemma-2-2b)|[📖](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)|
| [DCLM 7B](https://github.com/mlfoundations/dclm) | 7B | Apple | 2024.7.18 | [🤗](https://huggingface.co/apple/DCLM-7B) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.11794) |
| [Phi-3](https://github.com/microsoft/Phi-3CookBook/blob/main/README.md) | 3.8B 7B | Microsoft | 2024.4.23 | [🤗](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.14219)|
| Mistral NeMo | 12B | Mistral AI | 2024.6.18 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/mistral-nemo/) |
| [Gemma](https://github.com/google-deepmind/gemma) | 2B 7B | Google | 2024.2.21 |  [🤗](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)| [📖](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) |
| [Mistral 7B](https://github.com/mistralai/mistral-inference) | 2B 7B | Mistral AI | 2023.9.27 |  [🤗](https://huggingface.co/mistralai)| [📖](https://mistral.ai/news/announcing-mistral-7b/) |


**Embodied Model**





## LLM Inference 

|Title|Date|Org|Paper|
|:---:|:---:|:---:|:---:|
| [DashInfer-VLM](https://github.com/modelscope/dash-infer) | 2025.1 | ModelScope | [📖](https://dashinfer.readthedocs.io/en/latest/vlm/vlm_offline_inference_en.html) |
| SparseInfer | 2024.11 | University of Seoul, etc |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2411.12692) |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | 2024.6 | Moonshot AI | [📖](https://flashinfer.ai/2024/02/02/cascade-inference.html) |
| [flashinfer](https://github.com/flashinfer-ai/flashinfer) | 2024.2 | flashinfer-ai | [📖](https://flashinfer.ai/2024/02/02/cascade-inference.html)|
| [inferflow](https://github.com/inferflow/inferflow) | 2024.2 | Tencent AI Lab | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2401.08294.pdf) | 
| [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | 2023.12 | SJTU |  |
| [PETALS](https://github.com/bigscience-workshop/petals) | 2023.12 | HSE University, etc | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.08361.pdf) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)  | 2023.10 | NVIDIA | - |
| [LightSeq](https://github.com/RulinShao/LightSeq) | 2023.10 | UC Berkeley, etc | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2310.03294.pdf)  |
| [vLLM](https://github.com/vllm-project/vllm) | 2023.9 | UC Berkeley, etc | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2309.06180.pdf) | [📖](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf) |
| [StreamingLLM](https://github.com/mit-han-lab/streaming-llm) | 2023.9 | Meta AI, etc | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2309.17453.pdf)  |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | 2023.5 | mlc-ai | [📖](https://llm.mlc.ai/)  |
| [Medusa](https://github.com/FasterDecoding/Medusa)  | 2023.9 | Tianle Cai, etc |  [📖](https://sites.google.com/view/medusa-llm) | 
| [LightLLM](https://github.com/ModelTC/lightllm) | 2023.8 | ModelTC | - |
| FastServe | 2023.5 | Peking University | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.05920.pdf)
| [SpecInfer](https://github.com/flexflow/FlexFlow/tree/inference)  | 2023.05 | Peking University, etc |[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.09781.pdf) |
| [Ollama](https://github.com/ollama/ollama) | 2023.8 | Ollama Inc | - | 
| [LMDeploy](https://github.com/InternLM/lmdeploy) | 2023.6 | InternLM | [📖](https://lmdeploy.readthedocs.io/en/latest/) |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)| 2020.5 | NVIDIA | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/1909.08053.pdf) | 




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

| GPU Specs               | GeForce RTX 4080 Super | GeForce RTX 4070 Ti Super | GeForce RTX 4070 Super |
|:-------------------------:|:------------------------:|:-------------------------:|:-------------------------:|
| CUDA Cores             | 10,240                | 8448                     | 7168                  |
| Memory Configuration   | 16 GB GDDR6X          | 16 GB GDDR6X              | 12 GB GDDR6X           |
| Memory Interface Width | 256-bit               | 256-bit                   | 256-bit                |
| Memory Bandwidth       | 736 GB/s              | 736 GB/s                  | 736 GB/s               |
| Base Clock (GHz)       | 2.21 GHz              | 2.31 GHz                  | 1.92 GHz               |
| Boost Clock (GHz)      | 2.55 GHz              | 2.61 GHz                  | 2.48 GHz               |
| Graphics Card Power    | 320W                  | 285W                      | 200W                   |
| Recommended PSU        | 750W                  | 700W                      | 650W                   |
| Price     | $999                  | $799                      | $599                   |




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
| Price      | $1,599                | $1,199                | $799                  | $599                 | $399 (8GB)<br>$499 (16GB) | $299            |




## Hardware Applications

### AI Glasses
| Name  | Company | Model | Time  |  Price |
|:--:|:--:|:-----------:|:---------------:|:---------------:|
| [雷鸟V3](https://mp.weixin.qq.com/s/NBn81ocRqLhDKmtGXZuX0w) | 雷鸟创新 | Qwen | 2025.1.7 | ¥ 1799 + | 
| [闪极拍拍镜](https://mp.weixin.qq.com/s/exweWfZm1Eoc2LWM8tFsUg) | 闪极科技 | Qwen Kimi GLM, etc. | 2024.12.19 | ¥999 + |
| INMO GO2 | 影目科技 | - | 2024.11.29 | ¥3999 |
| Rokid Glasses | Rokid | Qwen | 2024.11.18 | ¥2499 |
| Looktech | Looktech | ChatGPT Claude Gemini | 2024.11.16 | $199 | 
| [Ray-Ban](https://www.ray-ban.com/usa) | Meta | Meta AI | 2023.9 | $299 | 






### Reference
[Awesome-LLMs-on-device](https://github.com/NexaAI/Awesome-LLMs-on-device)

[Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)

[数字生命卡兹克- AI硬件大全](https://datakhazix.feishu.cn/wiki/Zfp6wzb8eivwMqkSNgLcuiExnJd) 










