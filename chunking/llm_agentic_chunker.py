import re
from .base_chunker import BaseChunker
from .chunking import RecursiveTokenChunker
from utils.utils import openai_token_count
from llms.cohere_llm import CohereLLM
from tqdm import tqdm  # Import tqdm trực tiếp để tránh import mỗi lần chạy vòng lặp


class LLMAgenticChunkerv2(BaseChunker):
    def __init__(self, llm: CohereLLM):
        self.client = llm
        self.splitter = RecursiveTokenChunker(
            chunk_size=50,  # Giảm chunk_size để Cohere dễ nhận diện chủ đề hơn
            chunk_overlap=5,  # Thêm overlap giúp cải thiện phân tách
            length_function=openai_token_count
        )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant trained to split text into semantically meaningful sections. "
                    "Each section should contain information about a specific topic. "
                    "Your task is to analyze the provided text and identify where topic shifts occur. "
                    "Return 'split_after: X, Y' where X and Y are the chunk numbers after which a split should occur. "
                    "Ensure that the response includes **at least two split points**."
                )
            },
            {
                "role": "user",
                "content": (
                        f"CHUNKED_TEXT: {chunked_input}\n\n"
                        f"Identify topic transitions and return the chunk numbers where a split should be made. "
                        f"The response should be in ascending order and contain at least 2 split points."
                        + (
                            f"\nThe previous response '{invalid_response}' was invalid. Do not repeat these numbers. Try again."
                            if invalid_response
                            else ""
                        )
                )
            },
        ]
        return messages

    def split_text(self, text):
        chunks = self.splitter.split_text(text)
        split_indices = []
        current_chunk = 0
        total_chunks = len(chunks)

        if total_chunks == 1:
            print("⚠️ Only one chunk detected. Skipping Cohere API call.")
            return chunks

        progress_bar = tqdm(total=total_chunks, desc="📌 Processing chunks", dynamic_ncols=True)

        while current_chunk < total_chunks - 1:
            token_count = 0
            chunked_input = ""

            for i in range(current_chunk, total_chunks):
                token_count += openai_token_count(chunks[i])
                chunked_input += f"<|start_chunk_{i + 1}|>{chunks[i]}<|end_chunk_{i + 1}|>"
                if token_count > 800:
                    break

            messages = self.get_prompt(chunked_input, current_chunk)

            while True:
                try:
                    result_string = self.client.create_agentic_chunker_message(
                        system_prompt=messages[0]["content"],
                        messages=messages[1:],
                        max_tokens=200,
                        temperature=0.2
                    )

                    print(f"🔥 Cohere Response: {result_string}")  # Debug kết quả từ Cohere

                    split_after_line = [line for line in result_string.split("\n") if "split_after:" in line]
                    if not split_after_line:
                        print("⚠️ No valid split points detected. Using default behavior.")
                        break

                    numbers = list(map(int, re.findall(r"\d+", split_after_line[0])))

                    if numbers == sorted(numbers) and all(number >= current_chunk for number in numbers):
                        split_indices.extend(numbers)
                        break
                    else:
                        messages = self.get_prompt(chunked_input, current_chunk, numbers)
                except Exception as e:
                    print(f"⚠️ Error during chunking: {e}, retrying...")

            current_chunk = numbers[-1] if numbers else total_chunks
            progress_bar.update(current_chunk - progress_bar.n)

        progress_bar.close()

        chunks_to_split_after = [i - 1 for i in split_indices]
        docs = []
        current_chunk_text = ""

        for i, chunk in enumerate(chunks):
            current_chunk_text += chunk + " "
            if i in chunks_to_split_after:
                docs.append(current_chunk_text.strip())
                current_chunk_text = ""

        if current_chunk_text:
            docs.append(current_chunk_text.strip())

        return docs
