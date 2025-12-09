import queue
import threading
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
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
                dtype=torch.float16,
                trust_remote_code=True,
            )
            self.llm.eval() # When model should learn comment out!!!!
            print("LLM model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load LLM model. {e}")
            self.llm = None
            self.tokenizer = None

    def _stream_response(self, user_input: str):
        if not self.llm:
            return "ERROR: LLM not loaded."
        
        with self.llm_lock:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenizer=True,
                add_generation_prompt=True,
                return_tensors=None,
            )

            try:
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.llm.device)
                attention_mask = torch.ones_like(input_ids).to(self.llm.device)

                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_special_tokens=True,
                    decode_with_prefix_space=True,
                    skip_prompt=True,
                )

                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=streamer,
                )

                t = threading.Thread(target=self.llm.generate, kwargs=gen_kwargs)
                t.start()

                buf = ""
                for chunk in streamer:
                    buf += chunk
                    while " " in buf:
                        word, buf = buf.split(" ", 1)
                        if word:
                            yield (word, False)
                
                t.join()

                if buf.strip():
                    yield (buf.strip(), False)

                yield ("", True)
            
            except Exception as e:
                print(f"LLM stream error: {e}")
                yield ("Error: Failed to generate response.", True)

    def process_loop(self):
        print("LLM processor ready...")

        while True:
            user_input = self.prompt_queue.get()

            if user_input is None:
                break

            for word, is_final in self._stream_response(user_input):
                self.response_queue.put((word, is_final))

            self.prompt_queue.task_done()