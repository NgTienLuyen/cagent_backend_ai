import re
from typing import Any, List, Optional
from .base_chunker import BaseChunker
from .fixed_token_chunker import TextSplitter

# chia văn bản thành các đoạn nhỏ hơn dựa trên biểu thức chính quy (regex) và tùy chọn giữ lại hoặc loại bỏ các kí tự phân tách
def _split_text_with_regex (
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # chia doan van than cac doan nho hon
    '''
     text: str  Đoạn văn bản cần tách
     separator: str   Kí tự phân tách mà bạn sẽ dùng để chia nhỏ văn bản
     keep_separator: bool - Chỉ định xem có dữ lại kí tự phân tách trong các phần đã chia hay không
    '''
    if separator:
        if keep_separator:
            #
            _splits = re.split(f"{separator}", text)
            splits = [_splits +_splits[i+1] for i in range (1, len((_splits)),2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] +splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveTokenChunker(TextSplitter):
    ''' chia một văn bản lớn thành các đoạn nhỏ (chunks) mà không làm mất mạch nội dung'''
    def __init__(
            self,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            separators: Optional[List[str]] = None, # Danh sach ki tu phan tach uu tien
            keep_separator: bool = True,  # giữ lại ký tự phân tách trong kết quả hay không
            is_separator_regex: bool = False,  # kí tự phân tách có phải là regex không
            **kwargs: Any,  # các tham số khác
    ) -> None:
        """
                Khởi tạo bộ chia nhỏ văn bản.

                - Mặc định thử tách theo các ký tự xuống dòng, dấu chấm, dấu câu,...
                - Nếu không có ký tự phân tách phù hợp, sẽ tách từng ký tự một.
        """
        super().__init__(chunk_size = chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator, **kwargs)
        # nếu không có danh sách kí tự phân tách, sử dụng mặc định
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        self._is_separator_regex = is_separator_regex  # Xác định có sử dụng regex không

    def _split_text (self, text: str, separator: List[str]) -> List[str]:
        """Chia nhỏ văn bản bằng cách thử nhiều kí tự phân tách khác nhau
            - Bắt đầu với kí tự ưu tiên cao nhất trong danh sách
            - Nếu không tìm thấy kí tự phân tách phù hợp, tiếp tục thử cách tiếp theo.
            - Nếu không còn lựa chọn nào, chia thành kí tự riêng lẻ

            Args:
                text (str): Văn bản cần chia nhỏ.
                separators (List[str]): Danh sách ký tự phân tách.
            Returns:
                List[str]: Danh sách các đoạn văn bản đã chia nhỏ.
        """
        final_chunks = [] #Danh sach luu tru ket qua
        separator = separator[-1] # măc định lựa chọn kí tự phân tách cuối cùng
        new_separators = [] # Danh sách các kí tự phân tách tiếp theo cần

        # duyệt danh sách các kí tự phân tách để tìm cái phù hợp
        for i, _s in enumerate(separator):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            # kí tự phân tách rỗng -> chọn luôn
            if _s == "":
                separator = _s
                break;

            # Nếu tìm thấy kí tự phân tách trong văn bản, chọn nó
            if re.search(_separator, text):
                separator = _s
                new_separator = separator[i+1] #lưu lại danh sách còn lại
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        _good_splits = []  # Danh sách các đoạn phù hợp
        _separator = "" if self._keep_separator else separator  # Xác định có giữ lại ký tự phân tách không

        for s in splits:
            # Nếu đoạn này nhỏ hơn giới hạn, thêm vào danh sách tạm
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                # Nếu đã có các đoạn tạm, gộp chúng lại trước khi tiếp tục
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []

                # Nếu không còn ký tự phân tách khác, giữ nguyên đoạn văn bản dài
                if not new_separators:
                    final_chunks.append(s)
                else:
                    # Nếu còn ký tự phân tách khác, tiếp tục chia nhỏ đệ quy
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

            # Xử lý phần còn lại trong danh sách tạm
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Gọi `_split_text()` để thực hiện quá trình chia nhỏ văn bản.
        Args:
            text (str): Văn bản cần chia nhỏ.
        Returns:
            List[str]: Danh sách các đoạn văn bản đã chia nhỏ.
        """

        return self._split_text(text, self._separators)
