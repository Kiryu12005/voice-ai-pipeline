import queue
import threading

from llama_cpp import Llama
from ttt.config import MODEL_PATH, SYSTEM_PROMPT

class LLMCore:
    def __init__(self, prompt_queue: queue.Queue, response_queue: queue.Queue):
        self.prompt_queue = prompt_queue
        self.response_queue = response_queue
        self.llm_lock = threading.Lock()

        print(f"Loading LLM model from {MODEL_PATH}...")
        try:
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False,
            )
            print("LLM model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load LLM model. {e}")
            self.llm = None

    def _generate_response(self, user_input: str) -> str:
        if not self.llm:
            return "Error: LLM not loaded."
        
        with self.llm_lock:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]

            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7,
                    stop=["<|eot_id|>", "\n"]
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"LLM error: {e}")
                return "Error: Failed to generate response."
            
    def process_loop(self):
        print("LLM processor ready...")

        while True:
            user_input = self.prompt_queue.get()

            if user_input is None:
                break

            response = self._generate_response(user_input)
            self.response_queue.put(response)

            self.prompt_queue.task_done()