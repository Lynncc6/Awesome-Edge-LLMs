# 🔍 Awesome Edge LLMs 
A comprehensive survey on Edge AI，covering hardware, software, frameworks, applications, performance optimization, and the deployment of LLMs on edge devices.


## Open Source Edge Models

The listed models are base model limited to either of the following:
- Parameter ≤ 10B
- Officially claimed edge models



| Model           | Size | Org | Time  | Download | Paper |
|:-----------:|:--:|:--:|:-----------:|:---------------:|:---------------:|
| [VITA-1.5](https://github.com/VITA-MLLM/VITA) | 7B | VITA | 2025.1.6 | - | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2501.01957) |
| [InternVL 2.5](https://github.com/OpenGVLab/InternVL) | 8B | Shanghai AI Lab | 2024.12.22 | [🤗](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c) | - | 
| OmniAudio | 2.6B | Nexa AI | 2024.12.12 | [🤗](https://huggingface.co/NexaAIDev/OmniAudio-2.6B) | [📖](https://nexa.ai/blogs/omniaudio-2.6b) | 
| Phi-4 | 14B | Microsoft | 2024.12.12 | - | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2412.08905) |
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

|Date|Title|Paper|Code|
|:---:|:---:|:---:|:---:|
| 2024.11 | [**SparseInfer**] SparseInfer: Training-free Prediction of Activation Sparsity for Fast LLM Inference (@University of Seoul, etc) | [[pdf]](https://arxiv.org/pdf/2411.12692) | ⚠️ |
| 2024.08 | [**Decentralized LLM**] Decentralized LLM Inference over Edge Networks with Energy Harvesting (@Padova) | [[pdf]](https://arxiv.org/pdf/2408.15907) | ⚠️ |
| 2024.08 | [NanoFlow] NanoFlow: Towards Optimal Large Language Model Serving Throughput (@University of Washington) | [[pdf]](https://arxiv.org/pdf/2408.12757) | [![Stars](https://img.shields.io/github/stars/efeslab/Nanoflow.svg?style=social)](https://github.com/efeslab/Nanoflow) |
| 2024.07 | [DynamoLLM] DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency (@Microsoft Azure Research) | [[pdf]](https://arxiv.org/pdf/2408.00741) | ⚠️ |
| 2024.06 | [**Mooncake**] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (@Moonshot AI) | [[pdf]](https://github.com/kvcache-ai/Mooncake/blob/main/Mooncake-v1.pdf) | [![Stars](https://img.shields.io/github/stars/kvcache-ai/Mooncake.svg?style=social)](https://github.com/kvcache-ai/Mooncake) |
| 2024.02 | [**flashinfer**] FlashInfer: Kernel Library for LLM Serving (@flashinfer-ai) | [[docs]](https://flashinfer.ai/2024/02/02/cascade-inference.html) | [![Stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer.svg?style=social)](https://github.com/flashinfer-ai/flashinfer) |
| 2024.01 | [inferflow] INFERFLOW: AN EFFICIENT AND HIGHLY CONFIGURABLE INFERENCE ENGINE FOR LARGE LANGUAGE MODELS (@Tencent AI Lab) | [[pdf]](https://arxiv.org/pdf/2401.08294.pdf) | [![Stars](https://img.shields.io/github/stars/inferflow/inferflow.svg?style=social)](https://github.com/inferflow/inferflow) |
| 2023.12 | [PowerInfer] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU (@SJTU) | [[pdf]](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf) | [![Stars](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer.svg?style=social)](https://github.com/SJTU-IPADS/PowerInfer) |
| 2023.12 | [**PETALS**] Distributed Inference and Fine-tuning of Large Language Models Over The Internet (@HSE University, etc) | [[pdf]](https://arxiv.org/pdf/2312.08361.pdf) | [![Stars](https://img.shields.io/github/stars/bigscience-workshop/petals.svg?style=social)](https://github.com/bigscience-workshop/petals) |
| 2023.10 | [**TensorRT-LLM**] NVIDIA TensorRT LLM (@NVIDIA) | [[docs]](https://nvidia.github.io/TensorRT-LLM/) | [![Stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social)](https://github.com/NVIDIA/TensorRT-LLM) |
| 2023.10 | [LightSeq] LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers (@UC Berkeley, etc) | [[pdf]](https://arxiv.org/pdf/2310.03294.pdf) | [![Stars](https://img.shields.io/github/stars/RulinShao/LightSeq.svg?style=social)](https://github.com/RulinShao/LightSeq) |
| 2023.09 | [StreamingLLM] EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS (@Meta AI, etc) | [[pdf]](https://arxiv.org/pdf/2309.17453.pdf) | [![Stars](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social)](https://github.com/mit-han-lab/streaming-llm) |
| 2023.09 | [**vLLM**] Efficient Memory Management for Large Language Model Serving with PagedAttention (@UC Berkeley, etc) | [[pdf]](https://arxiv.org/pdf/2309.06180.pdf) | [![Stars](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social)](https://github.com/vllm-project/vllm) |
| 2023.09 | [Medusa] Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads (@Tianle Cai, etc) | [[blog]](https://sites.google.com/view/medusa-llm) | [![Stars](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg?style=social)](https://github.com/FasterDecoding/Medusa) |
| 2023.08 | [**LightLLM**] LightLLM is a Python-based LLM (Large Language Model) inference and serving framework (@ModelTC) | [[docs]](https://github.com/ModelTC/lightllm) | [![Stars](https://img.shields.io/github/stars/ModelTC/lightllm.svg?style=social)](https://github.com/ModelTC/lightllm) |
| 2023.06 | [**LMDeploy**] LMDeploy: LMDeploy is a toolkit for compressing, deploying, and serving LLMs (@InternLM) | [[docs]](https://lmdeploy.readthedocs.io/en/latest/) | [![Stars](https://img.shields.io/github/stars/InternLM/lmdeploy.svg?style=social)](https://github.com/InternLM/lmdeploy) |
| 2023.05 | [**MLC-LLM**] Universal LLM Deployment Engine with ML Compilation (@mlc-ai) | [[docs]](https://llm.mlc.ai/) | [![Stars](https://img.shields.io/github/stars/mlc-ai/mlc-llm.svg?style=social)](https://github.com/mlc-ai/mlc-llm) |
| 2023.05 | [**SpecInfer**] Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification (@Peking University, etc) | [[pdf]](https://arxiv.org/pdf/2305.09781.pdf) | [![Stars](https://img.shields.io/github/stars/flexflow/FlexFlow.svg?style=social)](https://github.com/flexflow/FlexFlow/tree/inference) |
| 2023.05 | [**FastServe**] Fast Distributed Inference Serving for Large Language Models (@Peking University, etc) | [[pdf]](https://arxiv.org/pdf/2305.05920.pdf) | ⚠️ |
| 2023.03 | [FlexGen] High-Throughput Generative Inference of Large Language Models with a Single GPU (@Stanford University, etc) | [[pdf]](https://arxiv.org/pdf/2303.06865.pdf) | [![Stars](https://img.shields.io/github/stars/FMInference/FlexGen.svg?style=social)](https://github.com/FMInference/FlexGen) |
| 2020.05 | [**Megatron-LM**] Training Multi-Billion Parameter Language Models Using Model Parallelism (@NVIDIA) | [[pdf]](https://arxiv.org/pdf/1909.08053.pdf) | [![Stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social)](https://github.com/NVIDIA/Megatron-LM) |


## Hardware Applications

### AI Glasses
| Name  | Company | Model | Time  |  Price |
|:--:|:--:|:-----------:|:---------------:|:---------------:|
| [Ray-Ban](https://www.ray-ban.com/usa) | Meta | Meta AI | 2023.9 | $299 | 
| 雷鸟V3 | 雷鸟创新 | Qwen | 2025.1.7 | ¥ 1799 + | 
| Rokid Glasses | Rokid | Qwen | 2024.11.18 | ¥2499 |
| 闪极拍拍镜 | 闪极科技 ｜ Qwen Kimi GLM etc. | 2024.12.19 ｜ ¥999 + ｜







References:
[数字生命卡兹克- AI硬件大全](https://datakhazix.feishu.cn/wiki/Zfp6wzb8eivwMqkSNgLcuiExnJd) @数字生命卡兹克










