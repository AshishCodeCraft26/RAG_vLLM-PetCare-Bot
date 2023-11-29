# RAG_vLLM PetCare Bot
<a href="https://blog.vllm.ai/2023/06/20/vllm.html" target="_blank" >vLLM</a> RAG Demo inside a Gradio app powered by BGE Embeddings, ChromaDB, and lmsys/vicuna-7b-v1.3.

Large language models (LLMs) currently face significant performance and resource challenges. They are slow, require extensive and expensive resources, and are limited by available GPU memory. To function effectively, these models need to fit entirely into video memory, along with a buffer for live inference sessions, which results in a low throughput of 1–3 requests per second even on expensive hardware.

To scale AI services efficiently, innovative solutions are necessary. Projects like vLLM are crucial in this context. vLLM introduces the PagedAttention algorithm, which allows LLMs to use non-contiguous blocks of memory during inference. This innovation enables more efficient use of video memory and supports a higher number of concurrent users on the same hardware.

The key challenges for generative AI in this regard are latency and throughput. The slow generation speed and low throughput raise infrastructure costs significantly. While hardware improvements over time will help, breakthroughs like vLLM are essential for deploying scalable AI services.


# vLLM: Fast and User-Friendly Library for LLM Inference and Serving

**vLLM** is a fast and easy-to-use library for Large Language Model (LLM) inference and serving, offering state-of-the-art performance and flexibility.

## Fast Performance
- **State-of-the-art Serving Throughput**: Maximizes efficiency in handling requests.
- **PagedAttention**: Efficiently manages attention key and value memory.
- **Continuous Batching**: Streamlines processing of incoming requests.
- **Optimized CUDA Kernels**: Enhances performance on compatible hardware.

## Flexibility and Ease of Use
- **Seamless Integration** with popular Hugging Face models.
- **High-Throughput Serving** with various decoding algorithms, including parallel sampling, beam search, and more.
- **Tensor Parallelism Support**: Facilitates distributed inference.
- **Streaming Outputs**: For real-time response generation.
- **OpenAI-Compatible API Server**: Ensures easy integration and compatibility.


The **RAG_vLLM PetCare Bot** is a powerful tool designed to provide pet information by answering user queries using state-of-the-art language models and vector stores. This README will guide you through the setup and usage of the RAG_vLLM PetCare Bot.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Introduction
[Short introduction about the RAG_vLLM PetCare Bot]

## Prerequisites
Before you can start using the RAG_vLLM PetCare Bot, make sure you have the following prerequisites installed on your system:
- Python 3.9 or higher
- Required Python packages (you can install them using pip):
  - langchain
  - vLLM
  - gradio
  - sentence-transformers
  - chromadb
  - PyPDF2 (for PDF document loading)

## Installation
1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/NeuroNest26/RAG_vLLM-PetCare-Bot.git
    cd RAG_vLLM-PetCare-Bot
    ```
2. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows, use: venv\Scripts\activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Download the required language models and data. Please refer to the Langchain documentation for specific instructions on how to download and set up the language model and vector store.

5. Set up the necessary paths and configurations in your project.

## Getting Started
To get started with the RAG_vLLM PetCare Bot, you need to:
1. Set up your environment and install the required packages as described in the [Installation](#installation) section.
2. Prepare the language model and data as per the Langchain documentation.
3. Start the bot by running the provided Python script or integrating it into your application.

## Usage
The RAG_vLLM PetCare Bot can be used for answering pet-related queries. To use the bot:
1. Start the bot by running your application or using the provided Python script.
2. Send a pet-related query to the bot.
3. The bot will provide a response based on the information available in its database.
4. If sources are found, they will be provided alongside the answer.
5. The bot can be customized to return specific information based on the query and context provided.


## License
This project is licensed under the MIT License.

For more information on how to use, configure, and extend the RAG_vLLM PetCare Bot, please refer to the Langchain documentation or contact the project maintainers.

Happy coding with RAG_vLLM PetCare Bot! 🚀


  







