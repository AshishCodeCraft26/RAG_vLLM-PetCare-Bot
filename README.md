# RAG_vLLM PetCare Bot
<a href="https://blog.vllm.ai/2023/06/20/vllm.html" target="_blank" >vLLM</a> RAG Demo inside a Gradio app powered by BGE Embeddings, ChromaDB, and lmsys/vicuna-7b-v1.3.

Large language models (LLMs) currently face significant performance and resource challenges. They are slow, require extensive and expensive resources, and are limited by available GPU memory. To function effectively, these models need to fit entirely into video memory, along with a buffer for live inference sessions, which results in a low throughput of 1–3 requests per second even on expensive hardware.

To scale AI services efficiently, innovative solutions are necessary. Projects like vLLM are crucial in this context. vLLM introduces the PagedAttention algorithm, which allows LLMs to use non-contiguous blocks of memory during inference. This innovation enables more efficient use of video memory and supports a higher number of concurrent users on the same hardware.

The key challenges for generative AI in this regard are latency and throughput. The slow generation speed and low throughput raise infrastructure costs significantly. While hardware improvements over time will help, breakthroughs like vLLM are essential for deploying scalable AI services.



