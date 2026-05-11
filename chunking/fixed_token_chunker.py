from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import tiktoken

from .base_chunker import BaseChunker

from dataclasses import dataclass

logger = logging.getLogger(__name__)
TS = TypeVar("TS", bound="TextSplitter")

class TextSplitter(BaseChunker, ABC):
    '''Interface chia van ban thanh nhieu phan'''

    '''
        chunk_size: Kích thước tối đa của các khối để trả về
        chunk_overlap: Trùng lặp ký tự giữa các khối
        length_function: Hàm đo độ dài của các khối đã cho
        keep_separator: Có giữ dấu phân cách trong các khối hay không
        add_start_index: Nếu `True`, bao gồm chỉ mục bắt đầu của khối trong dữ liệu
        strip_whitespace: Nếu `True`, xóa khoảng trắng khỏi đầu và cuối của mọi tài liệu    
    '''
    def __init__(
            self,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            length_function: Callable[[str], int] = len,
            keep_separator: bool = False,
            add_start_index: bool = False,
            strip_whitespace: bool = True,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Số token trùng lặp ({chunk_overlap}) lớn hơn chunk_size ({chunk_size}), cần nhỏ hơn."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace


    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        '''phương thức trừu tượng, mỗi lớp con cần triển khai phương thức này để chia nhỏ văn bản'''

    def _join_docs(self, docs: List[str], separator: str)->Optional[str]:
        # ghép danh sách các đoạn văn về thành 1 đoạn chuỗi duy nhất, sử dụng kí tự phân tách
        text = separator.join(docs)
        if self._strip_whitespace:  # loại bỏ khoảng trằng ở đầu và cuối
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """
            Hàm kết hợp các đoạn nhỏ thành các chunk lớn hơn với kích thước tối đa `chunk_size`,
            đảm bảo có phần trùng lặp giữa các chunk (dựa trên `chunk_overlap`).

            Args:
                splits (Iterable[str]): Danh sách các đoạn văn bản nhỏ.
                separator (str): Ký tự phân cách giữa các đoạn.

            Returns:
                List[str]: Danh sách các chunk văn bản đã kết hợp.
        """
        separation_len = self._length_function(separator) # độ dài kí tự phân tách

        docs = [] # danh sách chứa các chunk kết quả
        current_doc: List[str] = []  # Biến tạm chứa chunk hiện tại
        total = 0 # tổng độ dài của chunk hiện tại

        for d in splits:
            _len = self._length_function(d) # độ dài của đoạn hiện tại
            if total + _len + (separation_len
                                if len(current_doc)>0
                                else 0) > self._chunk_size:
                # nếu chunk đã tạo lớn hơn _chunk_size, log cảnh báo
                if total >self._chunk_size:
                    logger.warning(
                        f"Đã tạo một khối có kích thước {total},"
                        f"dài hơn {self._chunk_size} đã chỉ định"
                    )
                # nếu có dữ liệu trong current_doc thì ghép lại thành một chunk lớn hơn
                if len(current_doc)>0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # (sliding window) để duy trì phần trùng lặp giữa các chunk
                    while total >self._chunk_overlap or (
                        total + _len + (separation_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total >0
                    ):
                        total -= self._length_function(current_doc[0])+(
                            separation_len if len (current_doc) >1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separation_len if len(current_doc) >1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


    @classmethod
    def from_tiktoken_encoder(
            cls: Type[TS],
            encoding_name: str = "gpt2",
            model_name: Optional[str] = None, # Mô hình LLM cần dùng
            allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # các token đặc biệt được phép
            disallowed_special: Union[Literal["all"], Collection[str]] = "all", # các token đặc biệt bị loại bỏ
            **kwargs: Any,
    ) -> TS:
        """Tạo một TextSplitter sử dụng tiktoken encoder để tính độ dài."""

        try:
            import tiktoken  #thư viện tokenizer của openAI
        except ImportError:
            raise ImportError(
                "Không tìm thấy thư viện, hãy thử lại "
            )

        # Lấy tokenizer phù hợp với mô hình chỉ định
        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)  # Tokenizer theo mô hình cụ thể
        else:
            enc = tiktoken.get_encoding(encoding_name) # Tokenizer mặc định theo encoding


        # phương thức đếm số lượng token trong một văn bản
        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special = allowed_special,
                    disallowed_special = disallowed_special,
                )
            )

        # nếu là lớp hiện tại 'FixedTokenChunker' thì cần thêm một vài thông số đặc biệt
        if issubclass(cls, FixedTokenChunker):
            extra_kwargs =  {
                "encoding_name": encoding_name,
                "model_name": model_name,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs} # kết hợp kwargs với các thông số đặc biệt

        # trả về một instance của lớp với phương thức tính độ dài token
        return cls(length_function= _tiktoken_encoder, **kwargs)


class FixedTokenChunker(TextSplitter):
    """Lớp chia nhỏ các đoạn văn bản theo số lượng token, sử dụng tokenizer của mô hình."""

    def __init__(
            self,
            encoding_name: str = "cl100k_base", #Mã hóa mặc định dùng cho GPT4
            model_name: Optional[str] = None, # Tên mô hình
            chunk_size: int = 4000, # số lượng token tối đa trong mỗi đoạn (chunk)
            chunk_overlap: int = 200,
            allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",
            **kwargs: Any,
    ) -> None:
        '''Hàm khởi tạo FixedTokenChunker '''
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Không tìm thấy tiktoken. hãy cài đặt lại"
            )

        # nếu có model_name, lấy tokenizer của mô hình cụ thể
        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name) # tokenizer phu hop voi mo hinh
        else:
            enc = tiktoken.get_encoding(encoding_name) # su dung encoding mac dinh

        self._tokenizer = enc   # Luu tokenizer vao bien instance
        self._allowed_special = allowed_special    # luu danh sach token dac biet duoc phep
        self._disallowed_special = disallowed_special  # Luu danh sach token dac biet bi cam

    def split_text(self, text: str) -> List[str]:

        # Hàm encode: chuyển văn bản thành danh sách token(list[int])
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special= self._allowed_special, # token đặc biệt được phép giữ lại
                disallowed_special= self._disallowed_special, # token đặc biệt bị loại bỏ
            )

        # tạo một đối tượng Tokenizer
        tokenizer = Tokenizer(
            chunk_overlap= self._chunk_overlap,
            tokens_per_chunk= self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

@dataclass(frozen = True)
class Tokenizer:
    '''Lớp chứa các thông tin về các mã hóa và giải mã '''

    chunk_overlap: int #số token trùng lặp giữa các đoạn
    tokens_per_chunk: int # số token tối đa trong mỗi đoạn (chunk)
    decode: Callable[[List[int]], str] # Hàm chuyển danh sách token thành văn bản
    encode: Callable[[str], List[int]] # Hàm chuyển văn bản thành danh sách token

def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits







