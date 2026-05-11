import asyncio
import httpx
import time
import json

# --- Cấu hình bài test ---
URL = "http://localhost:8000/api/query/" # Đổi từ 127.0.0.1 sang localhost
TOTAL_REQUESTS = 2  # Tổng số request cần gửi
CONCURRENCY = 2     # Số lượng request gửi đồng thời (giả lập 10 người dùng)

# --- Payload mẫu (Lấy từ log của bạn) ---
PAYLOAD = {
    "knowledgeBaseId": "1a8e31c5-58b5-4c54-a962-d542b10a3024",
    "config_id": "fe19417e-19e7-46d5-bf5d-66cc7c49e26a",
    "query": "Giảng viên cơ hữu là gì",
    "chat_history": [],
    "user_id": "load_test_user"
}

async def send_request(client, session_id):
    """Gửi một request duy nhất và trả về status code."""
    try:
        response = await client.post(URL, json=PAYLOAD, timeout=180)
        if 200 <= response.status_code < 300:
            return "success"
        else:
            print(f"Lỗi request #{session_id+1}: {response.status_code}, {response.text}")
            return "failure"
    except Exception as e:
        print(f"Exception khi gửi request #{session_id+1}: {e}")
        return "failure"

async def main():
    """Hàm chính để chạy load test."""
    print("--- Bắt đầu Load Test ---")
    print(f"URL: {URL}")
    print(f"Tổng số request: {TOTAL_REQUESTS}")
    print(f"Số người dùng đồng thời: {CONCURRENCY}")
    print("-" * 25)

    results = []
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run_request(client, session_id):
        # Chờ semaphore trước khi gửi request
        async with semaphore:
            print(f"  -> Bắt đầu request #{session_id+1}...")
            result = await send_request(client, session_id)
            results.append(result)
            print(f"  <- Hoàn thành request #{session_id+1} ({result})")

    async with httpx.AsyncClient() as client:
        # Tạo và chạy tất cả các task
        tasks = [run_request(client, i) for i in range(TOTAL_REQUESTS)]
        await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    # --- In kết quả ---
    success_count = results.count("success")
    failure_count = results.count("failure")
    rps = success_count / total_time if total_time > 0 else 0

    print("\n--- Kết quả Load Test ---")
    print(f"Tổng thời gian: {total_time:.2f} giây")
    print(f"Số request thành công: {success_count}")
    print(f"Số request thất bại: {failure_count}")
    print(f"Requests Per Second (RPS): {rps:.2f}")
    print("-------------------------")

if __name__ == "__main__":
    asyncio.run(main()) 