#!/usr/bin/env python3
"""
Memory Monitor - Theo dõi memory usage của AI Agent Backend
Giúp debug và tối ưu memory consumption
"""

import psutil
import time
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

class MemoryMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.memory_history = []
        
    def get_system_memory(self) -> Dict:
        """Lấy thông tin memory của hệ thống"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_docker_memory(self) -> List[Dict]:
        """Lấy memory usage của Docker containers"""
        try:
            result = subprocess.run([
                "docker", "stats", "--no-stream", "--format", 
                "{{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
            ], capture_output=True, text=True, timeout=10)
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        container_name = parts[0]
                        cpu_percent = parts[1].replace('%', '')
                        mem_usage = parts[2]  # e.g., "1.2GiB / 2GiB"
                        mem_percent = parts[3].replace('%', '')
                        
                        # Parse memory usage
                        mem_parts = mem_usage.split(' / ')
                        used_mem = mem_parts[0] if len(mem_parts) > 0 else "0B"
                        limit_mem = mem_parts[1] if len(mem_parts) > 1 else "0B"
                        
                        containers.append({
                            "name": container_name,
                            "cpu_percent": float(cpu_percent) if cpu_percent else 0,
                            "memory_used": used_mem,
                            "memory_limit": limit_mem,
                            "memory_percent": float(mem_percent) if mem_percent else 0
                        })
            
            return containers
        except Exception as e:
            print(f"❌ Error getting Docker stats: {e}")
            return []
    
    def get_process_memory(self, process_name: str = "python") -> List[Dict]:
        """Lấy memory của các Python processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
            try:
                if process_name in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'main:app' in cmdline or 'uvicorn' in cmdline or 'gunicorn' in cmdline:
                        memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                        processes.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "memory_mb": round(memory_mb, 2),
                            "cmdline": cmdline[:100] + "..." if len(cmdline) > 100 else cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def analyze_memory_trend(self) -> Dict:
        """Phân tích xu hướng memory usage"""
        if len(self.memory_history) < 2:
            return {"trend": "insufficient_data", "change": 0}
        
        recent = self.memory_history[-1]
        older = self.memory_history[-2]
        
        change = recent["used_gb"] - older["used_gb"]
        if change > 0.5:
            trend = "increasing"
        elif change < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_gb": round(change, 2),
            "samples": len(self.memory_history)
        }
    
    def get_memory_recommendations(self, system_mem: Dict, docker_containers: List[Dict]) -> List[str]:
        """Đưa ra khuyến nghị tối ưu memory"""
        recommendations = []
        
        # Kiểm tra memory usage
        if system_mem["percent"] > 90:
            recommendations.append("🚨 CRITICAL: Memory usage > 90%. Consider reducing workers or skipping model preload")
        elif system_mem["percent"] > 80:
            recommendations.append("⚠️ WARNING: Memory usage > 80%. Monitor closely")
        
        # Kiểm tra Docker containers
        total_docker_mem = 0
        for container in docker_containers:
            if "ai-agent" in container["name"]:
                mem_percent = container["memory_percent"]
                total_docker_mem += mem_percent
                
                if mem_percent > 80:
                    recommendations.append(f"⚠️ Container {container['name']} using {mem_percent}% memory")
        
        # Khuyến nghị dựa trên available memory
        available_gb = system_mem["available_gb"]
        if available_gb < 2:
            recommendations.append("💡 LOW MEMORY: Use minimal config with 1 worker and skip model preload")
        elif available_gb < 4:
            recommendations.append("💡 MEDIUM MEMORY: Use local config with 2 workers")
        else:
            recommendations.append("💡 HIGH MEMORY: Can use production config with 4 workers")
        
        return recommendations
    
    def monitor_once(self) -> Dict:
        """Thực hiện một lần monitoring"""
        system_mem = self.get_system_memory()
        docker_containers = self.get_docker_memory()
        python_processes = self.get_process_memory()
        
        # Lưu vào history
        self.memory_history.append(system_mem)
        if len(self.memory_history) > 10:  # Giữ tối đa 10 samples
            self.memory_history.pop(0)
        
        trend = self.analyze_memory_trend()
        recommendations = self.get_memory_recommendations(system_mem, docker_containers)
        
        return {
            "system_memory": system_mem,
            "docker_containers": docker_containers,
            "python_processes": python_processes,
            "trend": trend,
            "recommendations": recommendations,
            "uptime_minutes": (datetime.now() - self.start_time).total_seconds() / 60
        }
    
    def print_status(self, data: Dict):
        """In trạng thái memory ra console"""
        print(f"\n{'='*60}")
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Uptime: {data['uptime_minutes']:.1f} minutes")
        print(f"{'='*60}")
        
        # System Memory
        sys_mem = data["system_memory"]
        print(f"💾 SYSTEM MEMORY:")
        print(f"   Total: {sys_mem['total_gb']} GB")
        print(f"   Used:  {sys_mem['used_gb']} GB ({sys_mem['percent']:.1f}%)")
        print(f"   Free:  {sys_mem['available_gb']} GB")
        
        # Trend
        trend = data["trend"]
        trend_icon = "📈" if trend["trend"] == "increasing" else "📉" if trend["trend"] == "decreasing" else "➡️"
        print(f"📊 TREND: {trend_icon} {trend['trend']} ({trend['change_gb']:+.2f} GB)")
        
        # Docker Containers
        if data["docker_containers"]:
            print(f"\n🐳 DOCKER CONTAINERS:")
            for container in data["docker_containers"]:
                print(f"   {container['name']}: {container['memory_used']} ({container['memory_percent']:.1f}%)")
        
        # Python Processes
        if data["python_processes"]:
            print(f"\n🐍 PYTHON PROCESSES:")
            for proc in data["python_processes"]:
                print(f"   PID {proc['pid']}: {proc['memory_mb']:.1f} MB - {proc['cmdline']}")
        
        # Recommendations
        if data["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in data["recommendations"]:
                print(f"   {rec}")
        
        print(f"{'='*60}")
    
    def run_continuous(self, interval: int = 30):
        """Chạy monitoring liên tục"""
        print("🚀 Starting Memory Monitor...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                data = self.monitor_once()
                self.print_status(data)
                
                # Lưu log file
                log_file = f"memory_monitor_{datetime.now().strftime('%Y%m%d')}.json"
                with open(log_file, 'a') as f:
                    f.write(json.dumps(data, indent=2) + '\n')
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Memory monitoring stopped")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Monitor for AI Agent Backend")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--once", "-o", action="store_true", help="Run once and exit")
    
    args = parser.parse_args()
    
    monitor = MemoryMonitor()
    
    if args.once:
        data = monitor.monitor_once()
        monitor.print_status(data)
    else:
        monitor.run_continuous(args.interval)

if __name__ == "__main__":
    main()