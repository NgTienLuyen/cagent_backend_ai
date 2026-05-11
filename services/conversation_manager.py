#!/usr/bin/env python3
"""
Conversation Manager - Quản lý conversation history để tối ưu token
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ConversationManager:
    """Quản lý conversation history với tối ưu token"""
    
    def __init__(self, max_history_messages: int = 10, max_history_age_hours: int = 24):
        """
        Args:
            max_history_messages: Số message tối đa trong history
            max_history_age_hours: Thời gian tối đa lưu history (giờ)
        """
        self.max_history_messages = max_history_messages
        self.max_history_age_hours = max_history_age_hours
        self._conversations: Dict[str, List[Dict]] = {}
        self._system_messages: Dict[str, Dict] = {}
        
        logger.info(f"[CONVERSATION_MANAGER] Khởi tạo với max_history={max_history_messages}, max_age={max_history_age_hours}h")
    
    def get_conversation_history(self, chat_section_id: str) -> List[Dict]:
        """Lấy conversation history cho chat_section_id"""
        if chat_section_id not in self._conversations:
            return []
        
        # Lọc messages cũ
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_age_hours)
        valid_messages = []
        
        for msg in self._conversations[chat_section_id]:
            msg_time = datetime.fromisoformat(msg.get('timestamp', '1970-01-01T00:00:00'))
            if msg_time > cutoff_time:
                valid_messages.append(msg)
        
        # Cập nhật conversation
        self._conversations[chat_section_id] = valid_messages
        
        # Giới hạn số messages
        if len(valid_messages) > self.max_history_messages:
            valid_messages = valid_messages[-self.max_history_messages:]
            self._conversations[chat_section_id] = valid_messages
        
        logger.info(f"[CONVERSATION_MANAGER] Lấy {len(valid_messages)} messages cho {chat_section_id}")
        return valid_messages
    
    def add_message(self, chat_section_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Thêm message vào conversation history"""
        if chat_section_id not in self._conversations:
            self._conversations[chat_section_id] = []
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self._conversations[chat_section_id].append(message)
        
        # Giới hạn số messages
        if len(self._conversations[chat_section_id]) > self.max_history_messages:
            self._conversations[chat_section_id] = self._conversations[chat_section_id][-self.max_history_messages:]
        
        logger.info(f"[CONVERSATION_MANAGER] Thêm {role} message cho {chat_section_id}")
    
    def set_system_message(self, chat_section_id: str, system_content: str, metadata: Optional[Dict] = None):
        """Lưu system message cho conversation"""
        self._system_messages[chat_section_id] = {
            'content': system_content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        logger.info(f"[CONVERSATION_MANAGER] Lưu system message cho {chat_section_id}")
    
    def get_system_message(self, chat_section_id: str) -> Optional[Dict]:
        """Lấy system message cho conversation"""
        return self._system_messages.get(chat_section_id)
    
    def build_optimized_messages(self, chat_section_id: str, user_content: str) -> List[Dict]:
        """
        Xây dựng messages tối ưu với conversation history
        
        Returns:
            List[Dict]: Messages format cho LLM
        """
        messages = []
        
        # 1. System message (nếu có)
        system_msg = self.get_system_message(chat_section_id)
        if system_msg:
            messages.append({
                'role': 'system',
                'content': system_msg['content']
            })
        
        # 2. Conversation history
        history = self.get_conversation_history(chat_section_id)
        messages.extend(history)
        
        # 3. User message hiện tại
        messages.append({
            'role': 'user',
            'content': user_content
        })
        
        logger.info(f"[CONVERSATION_MANAGER] Xây dựng {len(messages)} messages cho {chat_section_id}")
        return messages
    
    def has_conversation_history(self, chat_section_id: str) -> bool:
        """Kiểm tra xem có conversation history hay không"""
        history = self.get_conversation_history(chat_section_id)
        system_msg = self.get_system_message(chat_section_id)
        return len(history) > 0 or system_msg is not None
    
    def clear_conversation(self, chat_section_id: str):
        """Xóa conversation history"""
        if chat_section_id in self._conversations:
            del self._conversations[chat_section_id]
        if chat_section_id in self._system_messages:
            del self._system_messages[chat_section_id]
        logger.info(f"[CONVERSATION_MANAGER] Xóa conversation cho {chat_section_id}")
    
    def get_conversation_stats(self, chat_section_id: str) -> Dict:
        """Lấy thống kê conversation"""
        history = self.get_conversation_history(chat_section_id)
        system_msg = self.get_system_message(chat_section_id)
        
        return {
            'chat_section_id': chat_section_id,
            'history_count': len(history),
            'has_system_message': system_msg is not None,
            'oldest_message': history[0]['timestamp'] if history else None,
            'newest_message': history[-1]['timestamp'] if history else None
        }
    
    def cleanup_old_conversations(self):
        """Dọn dẹp conversations cũ"""
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_age_hours)
        removed_count = 0
        
        for chat_id in list(self._conversations.keys()):
            if not self._conversations[chat_id]:
                continue
                
            newest_msg = self._conversations[chat_id][-1]
            newest_time = datetime.fromisoformat(newest_msg.get('timestamp', '1970-01-01T00:00:00'))
            
            if newest_time < cutoff_time:
                self.clear_conversation(chat_id)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"[CONVERSATION_MANAGER] Đã dọn dẹp {removed_count} conversations cũ")

# Global instance
conversation_manager = ConversationManager()
