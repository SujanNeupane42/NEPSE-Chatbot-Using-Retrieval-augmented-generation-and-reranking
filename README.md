# NEPSE_ChatBot_LLM

This project makes use of open-source models to develop a chatbot for NEPSE, Nepal Stock Exchange Ltd, utilizing the Retrieval Augmented Generation technique. The public NEPSE booklet has been utilized for question-answering. This project utilizes the following open-source models:

1. Intel/neural-chat-7b-v3-1 -> This open-source LLM, originally developed by Intel, and quantized by TheBloke is utilized in this project. Specifically, the 8-bit GPTQ quantized version is utilized due to limited memory.
[Original Model](https://huggingface.co/Intel/neural-chat-7b-v3-1)
[Quantized Model](https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GPTQ)

 
3. all-mpnet-base-v2 -> An open-source sentence transformer from hugging face called all-mpnet-base-v2 is utilized to generate high-quality embeddings is utilized.
[Sentence Transformer](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
   
4. AAI/bge-reranker-large -> An open-source reranking model from hugging face called bge-reranker-large is utilized to re-rank the retrieved documents from vector store.
[Reranking](https://huggingface.co/BAAI/bge-reranker-large)

5. Google Translate API -> The free Google Translate API is utilized to perform translation between Nepali and English content.


