from datasets import load_dataset
import itertools

# Chỉ tải phần 'validation' của bộ dữ liệu NarrativeQA
validation_data = load_dataset("deepmind/narrativeqa", split="validation")

# Bây giờ biến 'validation_data' chỉ chứa dữ liệu của phần validation
# In ra để kiểm tra
print(validation_data)  

# In ra 2 dòng dữ liệu đầu tiên để xem nội dung
print("Hai dòng dữ liệu đầu tiên:")
for example in itertools.islice(validation_data, 1):
    print(example)
    print("-" * 20)