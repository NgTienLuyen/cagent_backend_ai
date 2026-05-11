import os
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import time
from openai import OpenAI

# Khởi tạo client OpenAI với API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Đảm bảo đã thiết lập OPENAI_API_KEY trong biến môi trường

# Sử dụng Assistant ID đã có sẵn
assistant_id = "asst_254Fy1EZeOcbbSsJFxEFsmak"


# Hàm để đợi assistant xử lý
def wait_for_run_completion(thread_id, run_id):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status == "completed":
            print("Trợ lý đang xử lý xong...")
            break
        elif run.status == "failed":
            print(f"Lỗi: {run.last_error}")
            break
        elif run.status == "requires_action":
            # Xử lý các yêu cầu hành động nếu assistant cần thực hiện tool calls
            print("Trợ lý đang yêu cầu thực hiện hành động...")
            required_actions = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for action in required_actions:
                # Ở đây bạn có thể xử lý các tool calls khác nhau
                # Đây là ví dụ đơn giản
                tool_outputs.append({
                    "tool_call_id": action.id,
                    "output": "Đây là kết quả từ tool call"
                })

            # Gửi kết quả của các tool calls
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run_id,
                tool_outputs=tool_outputs
            )
        else:
            print(f"Trợ lý đang xử lý... ({run.status})")
            time.sleep(1)  # Đợi 1 giây trước khi kiểm tra lại


# Hàm để gửi tin nhắn và nhận phản hồi
def chat_with_assistant(thread_id, assistant_id, user_message):
    # Gửi tin nhắn từ người dùng
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    # Chạy assistant để xử lý tin nhắn
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    # Đợi assistant xử lý
    wait_for_run_completion(thread_id, run.id)

    # Lấy tin nhắn mới nhất
    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
        order="desc",
        limit=1
    )

    if messages.data:
        try:
            assistant_message = messages.data[0].content[0].text.value
            return assistant_message
        except (AttributeError, IndexError) as e:
            print(f"Lỗi khi đọc phản hồi: {e}")
            return "Không thể đọc phản hồi từ assistant."
    else:
        return "Không nhận được phản hồi từ assistant."


# Hàm để lưu ID thread để sử dụng sau này
def save_thread_id_to_file(thread_id, filename="thread_id.txt"):
    with open(filename, "w") as f:
        f.write(thread_id)
    print(f"Đã lưu Thread ID vào {filename}")


# Hàm để tải thread ID từ tệp
def load_thread_id_from_file(filename="thread_id.txt"):
    try:
        with open(filename, "r") as f:
            thread_id = f.read().strip()
        print(f"Đã tải Thread ID từ {filename}")
        return thread_id
    except FileNotFoundError:
        print(f"Không tìm thấy tệp {filename}. Tạo thread mới.")
        return None


def main():
    # Lấy thông tin về assistant
    try:
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        print(f"Đã kết nối với Trợ lý: {assistant.name} (ID: {assistant.id})")
    except Exception as e:
        print(f"Lỗi khi kết nối với assistant: {e}")
        return

    # Kiểm tra xem có thread ID được lưu trước đó không
    print("Bạn muốn tiếp tục cuộc hội thoại cũ hay bắt đầu cuộc hội thoại mới?")
    print("1. Tiếp tục cuộc hội thoại cũ")
    print("2. Bắt đầu cuộc hội thoại mới")
    choice = input("Lựa chọn của bạn (1/2): ")

    if choice == "1":
        thread_id = load_thread_id_from_file()
        if not thread_id:
            print("Không tìm thấy cuộc hội thoại cũ. Bắt đầu cuộc hội thoại mới.")
            thread = client.beta.threads.create()
            thread_id = thread.id
            print(f"Đã tạo Thread mới với ID: {thread_id}")
    else:
        thread = client.beta.threads.create()
        thread_id = thread.id
        print(f"Đã tạo Thread mới với ID: {thread_id}")

    print("\n===== BẮT ĐẦU CUỘC HỘI THOẠI =====")
    print("Nhập 'exit' để thoát và lưu cuộc hội thoại")
    print("Nhập 'clear' để xóa và bắt đầu cuộc hội thoại mới")
    print("==============================\n")

    while True:
        user_input = input("\nBạn: ")

        if user_input.lower() == 'exit':
            # Lưu thread ID trước khi thoát
            save_thread_id_to_file(thread_id)
            print("Đã lưu cuộc hội thoại. Tạm biệt!")
            break

        if user_input.lower() == 'clear':
            thread = client.beta.threads.create()
            thread_id = thread.id
            print(f"Đã tạo Thread mới với ID: {thread_id}")
            continue

        # Gửi tin nhắn và nhận phản hồi
        print("Đang gửi tin nhắn...")
        response = chat_with_assistant(thread_id, assistant_id, user_input)
        print("\nTrợ lý:", response)


if __name__ == "__main__":
    main()