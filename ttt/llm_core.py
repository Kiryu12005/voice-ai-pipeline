import queue
import threading
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ttt.config import MODEL_PATH, SYSTEM_PROMPT

class LLMCore:
    def __init__(self, prompt_queue: queue.Queue, response_queue: queue.Queue):
        self.prompt_queue = prompt_queue
        self.response_queue = response_queue

        full_system_prompt = "\n".join(SYSTEM_PROMPT)
        self.system_prompt = full_system_prompt

        self.llm_lock = threading.Lock()

        print(f"Loading LLM model from {MODEL_PATH}...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

            self.llm = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.llm.eval() # When model should learn comment out!!!!
            print("LLM model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load LLM model. {e}")
            self.llm = None
            self.tokenizer = None

    def _generate_response(self, user_input: str) -> str:
        if not self.llm:
            return "Error: LLM not loaded."
        
        with self.llm_lock:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenizer=True,
                add_generation_prompt=True,
                returnn_tensors=None,
            )

            try:
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.llm.device)

                attention_mask = torch.ones_like(input_ids).to(self.llm.device)

                output = self.llm.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    # End token from llama3 chat model
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                response = self.tokenizer.decode(
                    output[0][input_ids.shape[-1]:],
                    skpip_special_tokens=True
                ).strip()

                return response
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