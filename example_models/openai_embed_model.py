import os
import pathlib
import pickle
import pathlib
import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv(verbose=True)

class OpenAIEmbedder:
    """
    Benchmark OpenAIs embeddings endpoint.
    """
    def __init__(self, engine, task_name=None, batch_size=128,  **kwargs):
        self.engine = engine
        self.max_token_len = 512
        self.batch_size = batch_size
        self.base_path = f"embeddings/{engine.split('/')[-1]}/"
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        self.task_name = task_name
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Add model_card_data for metadata extraction
        self.model_card_data = {
            "name": engine,
            "revision": None,
            "n_parameters": None,
            "memory_usage": None,
            "max_tokens": self.max_token_len,
            "embed_dim": 1536,  # OpenAI embeddings dimension
            "license": "proprietary",
            "open_source": False,
            "similarity_fn_name": "cosine",
            "framework": ["openai"]
        }

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
    
    def set_prompt(self, prompt: str):
        pass

    def encode(self, 
            sentences,
            decode=True,
            idx=None,
            **kwargs
        ):

        fin_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            if None in batch:
                for idx, e in enumerate(batch):
                    print("## ", idx, e)
                    raise ValueError("batch has None Value")
            
            batch = [e for e in batch if len(str(e).strip())>0]
            batch = [self.tokenizer.decode(
                self.tokenizer.encode(sentence)[:self.max_token_len]) 
                for sentence 
                in batch]
            
            out = [datum.embedding for datum in self.client.embeddings.create(input=batch, model=self.engine).data]

            fin_embeddings.extend(out)

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings
    