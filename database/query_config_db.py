# database/query_config_db.py
import json
import uuid
import traceback
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from .db_connection import get_pg_connection, return_pg_connection
from models.config_models import QueryConfigCreate, QueryConfigUpdate
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

thread_pool_executor = ThreadPoolExecutor(max_workers=10)

class QueryConfigDB:
    @staticmethod
    async def create_config(config: QueryConfigCreate) -> uuid.UUID:
        """Tạo cấu hình truy vấn mới"""
        conn, pool = None, None
        try:
            conn, pool = get_pg_connection()
            with conn.cursor() as cursor:
                # Nếu cấu hình mới là mặc định, đặt tất cả cấu hình khác của knowledge_base thành không mặc định
                if config.is_default:
                    cursor.execute(
                        'UPDATE query_config SET is_default = FALSE WHERE knowledge_base_id = %s',
                        (str(config.knowledge_base_id),)
                    )

                # Tiền xử lý các parameters để đảm bảo kiểu dữ liệu chính xác
                # Xử lý llm_config parameters
                if config.llm_config and hasattr(config.llm_config, 'parameters'):
                    llm_parameters = config.llm_config.parameters
                    for key in llm_parameters:
                        # Chuyển đổi các chuỗi số thành số thực
                        if isinstance(llm_parameters[key], str) and llm_parameters[key].replace('.', '', 1).isdigit():
                            if '.' in llm_parameters[key]:
                                llm_parameters[key] = float(llm_parameters[key])
                            else:
                                llm_parameters[key] = int(llm_parameters[key])
                
                # Xử lý prompt_builder parameters
                if config.prompt_builder and hasattr(config.prompt_builder, 'parameters'):
                    prompt_parameters = config.prompt_builder.parameters
                    for key in prompt_parameters:
                        # Các tham số đặc biệt cần chuyển đổi
                        if key in ['max_chunks', 'max_tokens']:
                            if isinstance(prompt_parameters[key], str) and prompt_parameters[key].isdigit():
                                prompt_parameters[key] = int(prompt_parameters[key])
                        elif key in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
                            if isinstance(prompt_parameters[key], str) and prompt_parameters[key].replace('.', '', 1).isdigit():
                                prompt_parameters[key] = float(prompt_parameters[key])
                        elif key in ['use_reranker'] and isinstance(prompt_parameters[key], str):
                            # Chuyển đổi chuỗi 'true'/'false' thành boolean
                            if prompt_parameters[key].lower() == 'true':
                                prompt_parameters[key] = True
                            elif prompt_parameters[key].lower() == 'false':
                                prompt_parameters[key] = False

                # Chuyển đổi llm_config và prompt_builder thành JSON
                llm_config_json = json.dumps(config.llm_config.dict())
                prompt_builder_json = json.dumps(config.prompt_builder.dict())

                cursor.execute(
                    """
                    INSERT INTO query_config 
                    (name_config, knowledge_base_id, llm_config, prompt_builder, is_default)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        config.name_config,
                        str(config.knowledge_base_id),
                        llm_config_json,
                        prompt_builder_json,
                        config.is_default
                    )
                )

                result = cursor.fetchone()
                conn.commit()
                return result[0]
        except Exception as e:
            logger.error(f"Error creating config: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                return_pg_connection(conn, pool)

    @staticmethod
    async def get_config(config_id: uuid.UUID):
        """Lấy thông tin cấu hình theo ID"""
        conn = None
        cursor = None
        try:
            logger.info(f"[DB] Lấy cấu hình với ID: {config_id}")
            conn, pool = get_pg_connection()
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT id, name_config, knowledge_base_id, "llm_config", prompt_builder, 
                           is_default, status, create_time
                    FROM query_config
                    WHERE id = %s AND is_deleted = FALSE
                    """,
                    (str(config_id),)
                )

                result = cursor.fetchone()
                if not result:
                    logger.warning(f"[DB] Không tìm thấy cấu hình với ID: {config_id}")
                    return None

                logger.info(f"[DB] Đã tìm thấy cấu hình với ID: {config_id}, name: {result[1]}")

                # Xử lý JSON một cách an toàn
                llm_config_data = result[3]
                prompt_builder_data = result[4]

                # Xử lý llm_config
                if isinstance(llm_config_data, str):
                    try:
                        llm_config = json.loads(llm_config_data)
                        logger.info(f"[DB] Đã đọc llm_config từ JSON")
                        
                        # Log thông tin quan trọng (che API key nếu có)
                        model_type = llm_config.get("model_type")
                        model_name = llm_config.get("model_name")
                        embedding_model = llm_config.get("embedding_model")
                        endpoint = llm_config.get("endpoint")
                        has_api_key = "api_key" in llm_config and llm_config["api_key"]
                        
                        logger.info(f"[DB] llm_config: model_type={model_type}, model_name={model_name}")
                        logger.info(f"[DB] llm_config: embedding_model={embedding_model}, endpoint={endpoint}")
                        logger.info(f"[DB] llm_config: has_api_key={'Có' if has_api_key else 'Không'}")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing llm_config JSON: {str(e)}")
                        llm_config = {}
                else:
                    llm_config = llm_config_data if llm_config_data is not None else {}

                # Xử lý prompt_builder
                if isinstance(prompt_builder_data, str):
                    try:
                        prompt_builder = json.loads(prompt_builder_data)
                        logger.info(f"[DB] Đã đọc prompt_builder từ JSON")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing prompt_builder JSON: {str(e)}")
                        prompt_builder = {}
                else:
                    prompt_builder = prompt_builder_data if prompt_builder_data is not None else {}

                # Xử lý create_time
                create_time = result[7]
                if create_time and hasattr(create_time, 'isoformat'):
                    create_time_str = create_time.isoformat()
                else:
                    create_time_str = str(create_time) if create_time is not None else None

                config_result = {
                    "id": result[0],
                    "name_config": result[1],
                    "knowledge_base_id": result[2],
                    "llm_config": llm_config,
                    "prompt_builder": prompt_builder,
                    "is_default": result[5],
                    "status": result[6],
                    "create_time": create_time_str
                }
                
                logger.info(f"[DB] Trả về cấu hình hoàn chỉnh cho ID: {config_id}")
                return config_result
        except Exception as e:
            logger.error(f"Error getting config {config_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                return_pg_connection(conn, pool)

    @staticmethod
    async def get_configs_by_knowledge_base(knowledge_base_id: uuid.UUID):
        """Lấy tất cả cấu hình của một knowledge base"""
        conn = None
        cursor = None
        try:
            logger.info(f"Getting configs for knowledge base ID: {knowledge_base_id}")

            conn, pool = get_pg_connection()
            with conn.cursor() as cursor:

                # Truy vấn chuẩn
                cursor.execute(
                    """
                    SELECT id, name_config, knowledge_base_id, llm_config, prompt_builder, 
                           is_default, status, create_time
                    FROM query_config
                    WHERE knowledge_base_id = %s AND is_deleted = FALSE
                    ORDER BY create_time DESC
                    """,
                    (str(knowledge_base_id),)
                )

                results = cursor.fetchall()
                logger.info(f"Found {len(results)} configs")

                configs = []
                for row in results:
                    try:
                        # Xử lý JSON một cách an toàn
                        llm_config_data = row[3]
                        prompt_builder_data = row[4]

                        # Xử lý llm_config
                        if isinstance(llm_config_data, str):
                            try:
                                llm_config = json.loads(llm_config_data)
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing llm_config JSON: {str(e)}")
                                llm_config = {}
                        else:
                            llm_config = llm_config_data if llm_config_data is not None else {}

                        # Xử lý prompt_builder
                        if isinstance(prompt_builder_data, str):
                            try:
                                prompt_builder = json.loads(prompt_builder_data)
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing prompt_builder JSON: {str(e)}")
                                prompt_builder = {}
                        else:
                            prompt_builder = prompt_builder_data if prompt_builder_data is not None else {}

                        # Xử lý create_time
                        create_time = row[7]
                        if create_time and hasattr(create_time, 'isoformat'):
                            create_time_str = create_time.isoformat()
                        else:
                            create_time_str = str(create_time) if create_time is not None else None

                        # Tạo object cấu hình
                        config = {
                            "id": row[0],
                            "name_config": row[1],
                            "knowledge_base_id": row[2],
                            "llm_config": llm_config,
                            "prompt_builder": prompt_builder,
                            "is_default": row[5] if row[5] is not None else False,
                            "status": row[6] if row[6] is not None else "active",
                            "create_time": create_time_str
                        }

                        configs.append(config)

                    except Exception as row_err:
                        logger.error(f"Error processing row: {str(row_err)}")
                        logger.error(traceback.format_exc())
                        continue

                return configs

        except Exception as e:
            logger.error(f"Database error in get_configs_by_knowledge_base: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                return_pg_connection(conn, pool)

    @staticmethod
    async def get_default_config(knowledge_base_id: uuid.UUID):
        """Lấy cấu hình mặc định của một knowledge base"""
        conn = None
        cursor = None
        try:
            logger.info(f"[DB] Lấy cấu hình mặc định cho knowledge base ID: {knowledge_base_id}")
            conn, pool = get_pg_connection()
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT id, name_config, knowledge_base_id, llm_config, prompt_builder, 
                           is_default, status, create_time
                    FROM query_config
                    WHERE knowledge_base_id = %s AND is_default = TRUE AND is_deleted = FALSE AND status = 'active'
                    LIMIT 1
                    """,
                    (str(knowledge_base_id),)
                )

                result = cursor.fetchone()
                if not result:
                    logger.warning(f"[DB] Không tìm thấy cấu hình mặc định cho knowledge base ID: {knowledge_base_id}")
                    return None

                logger.info(f"[DB] Đã tìm thấy cấu hình mặc định với ID: {result[0]}, name: {result[1]}")

                # Xử lý JSON một cách an toàn
                llm_config_data = result[3]
                prompt_builder_data = result[4]

                # Xử lý llm_config
                if isinstance(llm_config_data, str):
                    try:
                        llm_config = json.loads(llm_config_data)
                        logger.info(f"[DB] Đã đọc llm_config từ JSON")
                        
                        # Log thông tin quan trọng (che API key nếu có)
                        model_type = llm_config.get("model_type")
                        model_name = llm_config.get("model_name")
                        embedding_model = llm_config.get("embedding_model")
                        endpoint = llm_config.get("endpoint")
                        has_api_key = "api_key" in llm_config and llm_config["api_key"]
                        
                        logger.info(f"[DB] llm_config: model_type={model_type}, model_name={model_name}")
                        logger.info(f"[DB] llm_config: embedding_model={embedding_model}, endpoint={endpoint}")
                        logger.info(f"[DB] llm_config: has_api_key={'Có' if has_api_key else 'Không'}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing llm_config JSON: {str(e)}")
                        llm_config = {}
                else:
                    llm_config = llm_config_data if llm_config_data is not None else {}

                # Xử lý prompt_builder
                if isinstance(prompt_builder_data, str):
                    try:
                        prompt_builder = json.loads(prompt_builder_data)
                        logger.info(f"[DB] Đã đọc prompt_builder từ JSON")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing prompt_builder JSON: {str(e)}")
                        prompt_builder = {}
                else:
                    prompt_builder = prompt_builder_data if prompt_builder_data is not None else {}

                # Xử lý create_time
                create_time = result[7]
                if create_time and hasattr(create_time, 'isoformat'):
                    create_time_str = create_time.isoformat()
                else:
                    create_time_str = str(create_time) if create_time is not None else None

                config_result = {
                    "id": result[0],
                    "name_config": result[1],
                    "knowledge_base_id": result[2],
                    "llm_config": llm_config,
                    "prompt_builder": prompt_builder,
                    "is_default": result[5],
                    "status": result[6],
                    "create_time": create_time_str
                }
                logger.info(f"[DB] Trả về cấu hình mặc định hoàn chỉnh cho knowledge base ID: {knowledge_base_id}")
                return config_result
        except Exception as e:
            logger.error(f"Error getting default config for KB {knowledge_base_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                return_pg_connection(conn, pool)

    @staticmethod
    async def update_config(config_id: uuid.UUID, update_data: QueryConfigUpdate):
        """Cập nhật thông tin cấu hình"""
        conn = None
        cursor = None
        try:
            conn, pool = get_pg_connection()
            with conn.cursor() as cursor:

                # Lấy knowledge_base_id của config hiện tại
                cursor.execute(
                    'SELECT knowledge_base_id FROM query_config WHERE id = %s',
                    (str(config_id),)
                )
                result = cursor.fetchone()
                if not result:
                    return False

                knowledge_base_id = result[0]

                # Cập nhật các trường từ update_data
                update_fields = []
                params = []

                if update_data.name_config is not None:
                    update_fields.append('name_config = %s')
                    params.append(update_data.name_config)

                if update_data.llm_config is not None:
                    # Tiền xử lý parameters trước khi lưu
                    if hasattr(update_data.llm_config, 'parameters'):
                        llm_parameters = update_data.llm_config.parameters
                        for key in llm_parameters:
                            # Chuyển đổi các chuỗi số thành số thực
                            if isinstance(llm_parameters[key], str) and llm_parameters[key].replace('.', '', 1).isdigit():
                                if '.' in llm_parameters[key]:
                                    llm_parameters[key] = float(llm_parameters[key])
                                else:
                                    llm_parameters[key] = int(llm_parameters[key])
                    
                    update_fields.append('"llm_config" = %s')
                    params.append(json.dumps(update_data.llm_config.dict()))

                if update_data.prompt_builder is not None:
                    # Tiền xử lý parameters trước khi lưu
                    if hasattr(update_data.prompt_builder, 'parameters'):
                        prompt_parameters = update_data.prompt_builder.parameters
                        for key in prompt_parameters:
                            # Các tham số đặc biệt cần chuyển đổi
                            if key in ['max_chunks', 'max_tokens']:
                                if isinstance(prompt_parameters[key], str) and prompt_parameters[key].isdigit():
                                    prompt_parameters[key] = int(prompt_parameters[key])
                            elif key in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
                                if isinstance(prompt_parameters[key], str) and prompt_parameters[key].replace('.', '', 1).isdigit():
                                    prompt_parameters[key] = float(prompt_parameters[key])
                            elif key in ['use_reranker'] and isinstance(prompt_parameters[key], str):
                                # Chuyển đổi chuỗi 'true'/'false' thành boolean
                                if prompt_parameters[key].lower() == 'true':
                                    prompt_parameters[key] = True
                                elif prompt_parameters[key].lower() == 'false':
                                    prompt_parameters[key] = False
                    
                    update_fields.append('prompt_builder = %s')
                    params.append(json.dumps(update_data.prompt_builder.dict()))

                if update_data.status is not None:
                    update_fields.append('status = %s')
                    params.append(update_data.status)

                # Xử lý trường is_default riêng biệt
                if update_data.is_default is not None and update_data.is_default:
                    # Đặt tất cả cấu hình khác của knowledge_base thành không mặc định
                    cursor.execute(
                        'UPDATE query_config SET is_default = FALSE WHERE knowledge_base_id = %s',
                        (knowledge_base_id,)
                    )
                    update_fields.append('is_default = TRUE')
                elif update_data.is_default is not None:
                    update_fields.append('is_default = FALSE')

                if not update_fields:
                    return False

                # Thực hiện câu lệnh cập nhật
                params.append(str(config_id))
                cursor.execute(
                    f"""
                    UPDATE query_config
                    SET {", ".join(update_fields)}
                    WHERE id = %s
                    """,
                    params
                )

                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating config {config_id}: {str(e)}")
            logger.error(traceback.format_exc())
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                return_pg_connection(conn, pool)

    @staticmethod
    async def delete_config(config_id: uuid.UUID):
        """Xóa mềm cấu hình (đánh dấu là đã xóa)"""
        conn = None
        cursor = None
        try:
            conn, pool = get_pg_connection()
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    UPDATE query_config
                    SET is_deleted = TRUE, delete_time = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING is_default, knowledge_base_id
                    """,
                    (str(config_id),)
                )

                result = cursor.fetchone()
                if not result:
                    conn.commit()
                    return False

                # Nếu cấu hình bị xóa là mặc định, đặt một cấu hình khác làm mặc định
                is_default, knowledge_base_id = result
                if is_default:
                    cursor.execute(
                        """
                        UPDATE query_config
                        SET is_default = TRUE
                        WHERE knowledge_base_id = %s AND is_deleted = FALSE AND id != %s
                        LIMIT 1
                        """,
                        (knowledge_base_id, str(config_id))
                    )

                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting config {config_id}: {str(e)}")
            logger.error(traceback.format_exc())
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                return_pg_connection(conn, pool)
