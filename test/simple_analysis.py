#!/usr/bin/env python3
"""
Script phân tích dependencies đơn giản
"""

def parse_requirements(file_path):
    """Parse requirements file"""
    packages = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Lấy tên package (bỏ version)
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                    packages.add(package_name.lower())
    except FileNotFoundError:
        print(f"File {file_path} không tồn tại")
    return packages

def main():
    print("🔍 PHÂN TÍCH DEPENDENCIES")
    print("=" * 50)
    
    # Đọc các file requirements
    installed_packages = parse_requirements("full-requirements.txt")
    used_packages = parse_requirements("requirements.txt")
    
    print(f"📦 Tổng số thư viện đã cài: {len(installed_packages)}")
    print(f"📦 Số thư viện thực sự dùng: {len(used_packages)}")
    
    # Tìm các thư viện không dùng
    unused_packages = installed_packages - used_packages
    print(f"🧹 Số thư viện có thể xóa: {len(unused_packages)}")
    
    # Tìm các thư viện thiếu
    missing_packages = used_packages - installed_packages
    print(f"❌ Số thư viện thiếu: {len(missing_packages)}")
    
    print("\n" + "=" * 50)
    
    if unused_packages:
        print("🧹 CÁC THƯ VIỆN CÓ THỂ XÓA (top 20):")
        print("-" * 40)
        for i, pkg in enumerate(sorted(unused_packages)[:20]):
            print(f"  {i+1:2d}. {pkg}")
        
        if len(unused_packages) > 20:
            print(f"  ... và {len(unused_packages) - 20} thư viện khác")
    
    if missing_packages:
        print("\n❌ CÁC THƯ VIỆN THIẾU:")
        print("-" * 30)
        for pkg in sorted(missing_packages):
            print(f"  • {pkg}")
    
    # Tạo script PowerShell để gỡ cài đặt
    if unused_packages:
        print(f"\n💡 Tạo script gỡ cài đặt...")
        
        ps_script = """# Script PowerShell gỡ cài đặt các thư viện không dùng
# Chạy: .\\uninstall_unused.ps1

Write-Host "🧹 Gỡ cài đặt các thư viện không dùng..." -ForegroundColor Green

"""
        
        for pkg in sorted(unused_packages):
            ps_script += f'pip uninstall -y {pkg}\n'
        
        ps_script += """
Write-Host "✅ Hoàn thành gỡ cài đặt!" -ForegroundColor Green
"""
        
        with open("uninstall_unused.ps1", "w", encoding="utf-8") as f:
            f.write(ps_script)
        
        print("📝 Đã tạo: uninstall_unused.ps1")
    
    print("\n" + "=" * 50)
    print("🎯 KẾT LUẬN:")
    print("• Sử dụng requirements.txt mới (đã được pipreqs tạo)")
    print("• Chạy .\\uninstall_unused.ps1 để làm sạch môi trường")
    print("• Kiểm tra kỹ trước khi gỡ cài đặt")

if __name__ == "__main__":
    main() 