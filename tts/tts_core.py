import queue
import time

class TTSCore:
    def __init__(self, response_queue: queue.Queue):
        self.response_queue = response_queue

        print("TTS-Core initialized.")

    def process_responses(self):
        print("TTS-Core processing responses...")

        while True:
            llm_response = self.response_queue.get()

            if llm_response is None:
                break

            print(f"[JIJI SAYS]: {llm_response}")

            self.response_queue.task_done()
            time.sleep(0.1)