from pathlib import Path

# Tạo đường dẫn từ các phần
file_path = Path("D:/Semester_7-Internship/Chatbot_Project/CMC_chatbot_backend/aiagentsystem/uploads/file-1740646516185-593036289.docx")

if file_path.exists():
    print("Tệp đã tồn tại.")
else:
    print("Tệp không tồn tại.")
