#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库管理系统 - Flask Web界面
集成question2.py和question3.py的功能，提供用户友好的图形界面
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import os
import threading
import time
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import re
import uuid
import shutil
from question2 import HistoryManager  # 导入HistoryManager类

# 导入现有功能模块
try:
    from question3 import IndexUpdateManager, DirectoryMonitorHandler, FileChangeTracker
    from question2 import RAGSystem, DocumentProcessingPipeline, HistoryManager
    # 导入question1中的PDF文本提取函数
    from question1 import extract_text_from_pdf
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("确保question1.py、question2.py和question3.py在当前目录中")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("web_app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("WebApp")

# 配置watchdog日志
watchdog_logger = logging.getLogger("watchdog")
watchdog_logger.setLevel(logging.DEBUG)
watchdog_file_handler = logging.FileHandler("watchdog.log", encoding="utf-8")
watchdog_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
watchdog_logger.addHandler(watchdog_file_handler)
watchdog_logger.addHandler(logging.StreamHandler())

# 初始化Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'txts'
app.config['KB_FOLDER'] = 'knowledge_base'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 全局变量存储服务状态和结果
monitor_thread = None
index_manager = None
observer = None
is_monitoring = False
monitor_logs = []
MAX_LOGS = 100
system_status = {
    "start_time": None,
    "files_processed": 0,
    "last_file": "",
    "is_monitoring": False
}

# 文件类型限制
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md', 'xlsx', 'xls'}

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_log(message, level="INFO"):
    """添加日志到内存并限制数量"""
    global monitor_logs
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    monitor_logs.append({
        "time": timestamp,
        "level": level,
        "message": message
    })
    
    # 限制日志数量
    if len(monitor_logs) > MAX_LOGS:
        monitor_logs = monitor_logs[-MAX_LOGS:]
    
    # 记录到系统日志
    if level == "INFO":
        logger.info(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)

def start_monitor_service(data_dir=None, kb_dir=None, model_name=None, initial_scan=False):
    """
    启动文件监控服务
    
    Args:
        data_dir: 数据目录，默认为app.config['UPLOAD_FOLDER']
        kb_dir: 知识库目录，默认为app.config['KB_FOLDER']
        model_name: 向量模型名称，默认为'all-MiniLM-L6-v2'
        initial_scan: 是否执行初始扫描
        
    Returns:
        成功返回True，失败返回False
    """
    global monitor_thread, index_manager, observer, is_monitoring, system_status
    
    # 使用默认值
    data_dir = data_dir or app.config['UPLOAD_FOLDER']
    kb_dir = kb_dir or app.config['KB_FOLDER']
    model_name = model_name or 'all-MiniLM-L6-v2'
    
    try:
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(kb_dir, exist_ok=True)
        
        # 初始化索引管理器
        logger.info(f"====== 正在初始化索引管理器，知识库路径: {kb_dir} ======")
        add_log(f"正在初始化索引管理器，知识库路径: {kb_dir}")
        index_manager = IndexUpdateManager(
            knowledge_base_path=kb_dir,
            model_name=model_name
        )
        
        # 执行初始扫描（如果需要）
        if initial_scan:
            logger.info(f"====== 执行初始扫描: {data_dir} ======")
            add_log(f"执行初始扫描: {data_dir}")
            success = index_manager.rebuild_index(data_dir)
            if not success:
                logger.error("====== 初始扫描失败 ======")
                add_log("初始扫描失败", "ERROR")
                return False
        
        # 初始化监控处理器
        logger.info(f"====== 初始化文件监控处理器 ======")
        event_handler = DirectoryMonitorHandler(index_manager, data_dir)
        
        # 设置观察者
        logger.info(f"====== 设置文件系统观察者 ======")
        from watchdog.observers import Observer
        observer = Observer()
        observer.schedule(event_handler, data_dir, recursive=True)
        observer.start()
        
        is_monitoring = True
        system_status["is_monitoring"] = True
        system_status["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"====== 成功启动监控目录: {data_dir} ======")
        add_log(f"开始监控目录: {data_dir}")
        
        # 启动处理线程
        def process_changes():
            logger.info("====== 文件变更处理线程已启动 ======")
            while is_monitoring:
                try:
                    logger.debug("正在检查文件变更...")
                    event_handler.process_pending_changes()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"====== 监控线程出错: {str(e)} ======")
                    add_log(f"监控线程出错: {str(e)}", "ERROR")
            logger.info("====== 文件变更处理线程已停止 ======")
        
        monitor_thread = threading.Thread(target=process_changes)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return True
    except Exception as e:
        logger.error(f"====== 启动监控服务失败: {str(e)} ======")
        add_log(f"启动监控服务失败: {str(e)}", "ERROR")
        return False

def stop_monitor_service():
    """停止文件监控服务"""
    global observer, is_monitoring, system_status
    try:
        if observer:
            observer.stop()
            observer.join()
            add_log("文件监控服务已停止")
        
        is_monitoring = False
        system_status["is_monitoring"] = False
        return True
    except Exception as e:
        add_log(f"停止监控服务失败: {str(e)}", "ERROR")
        return False

def get_file_list():
    """获取监控目录中的文件列表"""
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        # 获取已索引文件的列表
        indexed_files = set()
        try:
            # 尝试读取索引配置
            documents_path = os.path.join(app.config['KB_FOLDER'], 'documents.json')
            if os.path.exists(documents_path):
                with open(documents_path, 'r', encoding='utf-8') as f:
                    try:
                        documents = json.load(f)
                        for doc in documents:
                            if 'metadata' in doc and 'source' in doc['metadata']:
                                source = doc['metadata']['source']
                                indexed_files.add(source)
                    except json.JSONDecodeError:
                        logger.warning("文档索引文件格式错误")
        except Exception as e:
            logger.error(f"获取索引文件列表失败: {str(e)}")
        
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if f.endswith('.txt') or f.endswith('.pdf'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
                modified_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                
                # 检查文件是否已索引
                is_indexed = f in indexed_files or is_monitoring  # 如果监控已启动，假定文件已索引
                
                files.append({
                    'name': f,
                    'size': round(os.path.getsize(file_path) / 1024, 2),  # KB
                    'modified': modified_time,
                    'indexed': is_indexed
                })
    return files

def get_system_status():
    """获取系统状态信息"""
    global system_status
    
    # 更新文件数量
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        file_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                      if f.endswith('.txt') or f.endswith('.pdf')])
        system_status["files_processed"] = file_count
    
    return system_status

# 路由定义
@app.route('/')
def index():
    """主页显示系统状态和基本操作"""
    status = get_system_status()
    return render_template('index.html', status=status, is_monitoring=is_monitoring)

@app.route('/search')
def search_page():
    """搜索界面"""
    return render_template('search.html')

@app.route('/files')
def files_page():
    """文件管理界面"""
    files = get_file_list()
    return render_template('files.html', files=files)

@app.route('/logs')
def logs_page():
    """系统日志界面"""
    global monitor_logs
    return render_template('logs.html', logs=monitor_logs)

# API路由
@app.route('/api/start_monitoring', methods=['POST'])
def api_start_monitoring():
    """启动监控服务API"""
    data = request.json or {}
    data_dir = data.get('data_dir', app.config['UPLOAD_FOLDER'])
    kb_dir = data.get('kb_dir', app.config['KB_FOLDER'])
    model_name = data.get('model_name', 'all-MiniLM-L6-v2')
    initial_scan = data.get('initial_scan', False)
    
    success = start_monitor_service(data_dir, kb_dir, model_name, initial_scan)
    return jsonify({'success': success})

@app.route('/api/stop_monitoring', methods=['POST'])
def api_stop_monitoring():
    """停止监控服务API"""
    success = stop_monitor_service()
    return jsonify({'success': success})

@app.route('/api/status', methods=['GET'])
def api_status():
    """获取系统状态API"""
    status = get_system_status()
    return jsonify(status)

@app.route('/api/logs', methods=['GET'])
def api_logs():
    """获取系统日志API"""
    global monitor_logs
    return jsonify(monitor_logs)

@app.route('/api/files', methods=['GET'])
def api_files():
    """获取文件列表API"""
    files = get_file_list()
    return jsonify({"files": files})

@app.route('/api/search', methods=['POST'])
def api_search():
    """执行搜索API"""
    data = request.json or {}
    query = data.get('query', '')
    
    if not query:
        return jsonify({'success': False, 'error': '查询内容为空'})
    
    try:
        # 使用RAGSystem执行查询
        add_log(f"执行查询: {query}")
        rag_system = RAGSystem(
            knowledge_base_path=app.config['KB_FOLDER'],
            api_key="c66be1dbcd07484fa81efde6d883e410.O2y7leHySzSP7Ygc"
        )
        
        result = rag_system.answer_query(query)
        
        # 转换结果为前端期望的格式
        retrieved_docs = []
        for doc in result.get('retrieved_documents', []):
            # 提取文件名
            source = doc.get('metadata', {}).get('source', '未知')
            # 提取文本内容的前200个字符作为预览
            text = doc.get('text', '')
            snippet = text[:200] + '...' if len(text) > 200 else text
            # 计算匹配分数（0-100之间）
            score = min(doc.get('score', 0) * 100, 100)
            
            retrieved_docs.append({
                'file_name': source,
                'snippet': snippet,
                'score': score,
                'text': text  # 完整文本用于详情查看
            })
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'relevant_documents': retrieved_docs
        })
    except Exception as e:
        add_log(f"搜索失败: {str(e)}", "ERROR")
        return jsonify({'success': False, 'error': f'搜索过程中出错: {str(e)}'})

@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    """上传文件API"""
    # 检查文件是否存在
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        try:
            # 保存原始文件名，只做最小限度处理以避免路径注入
            filename = file.filename.replace('/', '_').replace('\\', '_')
            
            # 确保上传目录存在
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # 如果是PDF文件，保存到原始位置并转换为TXT
            if filename.lower().endswith('.pdf'):
                # 先保存PDF文件到临时目录
                pdf_dir = 'pdfs'
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_dir, filename)
                
                logger.info(f"======= 开始保存PDF文件: {filename} =======")
                file.save(pdf_path)
                logger.info(f"======= PDF文件已保存: {pdf_path} =======")
                
                # 使用question1.py的函数提取文本并保存为TXT
                try:
                    logger.info(f"======= 开始从PDF提取文本: {filename} =======")
                    extract_text_from_pdf(pdf_path)
                    
                    # 生成TXT文件名 - 与PDF文件同名但扩展名不同
                    txt_filename = f"{os.path.splitext(filename)[0]}.txt"
                    logger.info(f"======= 文本提取成功，保存为TXT: {txt_filename} =======")
                    add_log(f"PDF文件已转换为TXT: {filename} -> {txt_filename}")
                    
                    # 更新filename变量为TXT文件名，以便后续处理
                    filename = txt_filename
                except Exception as e:
                    logger.error(f"======= 从PDF提取文本失败: {str(e)} =======")
                    add_log(f"PDF文本提取失败: {str(e)}", "ERROR")
                    return jsonify({'success': False, 'error': f'PDF文本提取失败: {str(e)}'})
            else:
                # 非PDF文件直接保存到上传目录
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logger.info(f"======= 开始保存文件: {filename} =======")
                file.save(file_path)
                logger.info(f"======= 文件已保存: {file_path} =======")
            
            add_log(f"文件已上传: {filename}")
            
            # 如果索引管理器存在，处理新文件
            if index_manager and is_monitoring:
                logger.info(f"======= 开始处理新文件: {filename} =======")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                success = index_manager.process_new_file(file_path)
                if success:
                    logger.info(f"======= 文件成功添加到索引: {filename} =======")
                    add_log(f"文件已添加到索引: {filename}")
                else:
                    logger.warning(f"======= 文件未能添加到索引: {filename} =======")
                    add_log(f"文件未能添加到索引: {filename}", "WARNING")
            else:
                logger.warning(f"======= 索引管理器未初始化或监控未启动，无法添加到索引: {filename} =======")
                add_log(f"索引管理器未初始化或监控未启动，无法添加到索引: {filename}", "WARNING")
            
            return jsonify({'success': True, 'filename': filename})
        except Exception as e:
            logger.error(f"======= 文件上传失败: {str(e)} =======")
            add_log(f"文件上传失败: {str(e)}", "ERROR")
            return jsonify({'success': False, 'error': f'文件上传失败: {str(e)}'})
    else:
        return jsonify({'success': False, 'error': '不支持的文件类型'})

@app.route('/api/file_details/<path:file_id>', methods=['GET'])
def api_file_details(file_id):
    """获取文件详情API"""
    try:
        # 使用原始文件名，只做最小限度处理以避免路径注入
        filename = file_id
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        
        stat = os.stat(file_path)
        modified_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # 读取文件预览内容
        preview = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                preview = f.read(2000)  # 读取前2000个字符作为预览
        except:
            preview = "无法预览此文件内容"
        
        # 检查文件是否已索引
        indexed = False
        if index_manager:
            indexed = True  # 简化处理，实际应该检查索引中是否包含该文件
        
        file_info = {
            'name': filename,
            'path': file_path,
            'size': stat.st_size,
            'modified': modified_time,
            'indexed': indexed,
            'preview': preview
        }
        
        return jsonify({'success': True, 'file': file_info})
    except Exception as e:
        return jsonify({'success': False, 'error': f'获取文件详情失败: {str(e)}'}), 500

@app.route('/api/delete_file/<path:file_id>', methods=['DELETE'])
def api_delete_file_by_id(file_id):
    """通过ID删除文件API"""
    try:
        # 使用原始文件名，只做最小限度处理以避免路径注入
        filename = file_id
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        
        # 如果索引管理器存在，处理文件删除
        if index_manager:
            if is_monitoring:
                success = index_manager.process_deleted_file(file_path)
                if success:
                    add_log(f"文件已从索引中移除: {filename}")
                else:
                    add_log(f"文件未能从索引中移除: {filename}", "WARNING")
            else:
                # 即使监控未启动，也记录这个删除操作
                add_log(f"监控未启动，但将在下次启动时重建索引: {filename}")
                # 尝试直接从本地进行索引重建
                try:
                    if os.path.exists(app.config['KB_FOLDER']):
                        # 临时初始化一个索引管理器并处理删除
                        temp_index_manager = IndexUpdateManager(
                            knowledge_base_path=app.config['KB_FOLDER'],
                            model_name='all-MiniLM-L6-v2'
                        )
                        success = temp_index_manager.rebuild_index(app.config['UPLOAD_FOLDER'])
                        if success:
                            add_log(f"文件删除后成功重建索引: {filename}")
                        else:
                            add_log(f"文件删除后重建索引失败: {filename}", "WARNING")
                except Exception as rebuild_error:
                    add_log(f"文件删除后尝试重建索引时出错: {str(rebuild_error)}", "ERROR")
        
        # 删除文件
        os.remove(file_path)
        add_log(f"文件已删除: {filename}")
        
        return jsonify({'success': True})
    except Exception as e:
        add_log(f"文件删除失败: {str(e)}", "ERROR")
        return jsonify({'success': False, 'error': f'文件删除失败: {str(e)}'}), 500

@app.route('/api/rebuild_index', methods=['POST'])
def api_rebuild_index():
    """重建索引API"""
    if not index_manager:
        return jsonify({'success': False, 'error': '索引管理器未初始化'})
    
    try:
        add_log("开始重建索引...")
        success = index_manager.rebuild_index(app.config['UPLOAD_FOLDER'])
        
        if success:
            add_log("索引重建成功")
            return jsonify({'success': True})
        else:
            add_log("索引重建失败", "ERROR")
            return jsonify({'success': False, 'error': '索引重建失败'})
    except Exception as e:
        add_log(f"索引重建时出错: {str(e)}", "ERROR")
        return jsonify({'success': False, 'error': f'索引重建时出错: {str(e)}'})

@app.route('/api/upload_files', methods=['POST'])
def api_upload_files():
    """上传多个文件API"""
    # 检查文件是否存在
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': '没有文件'}), 400
    
    files = request.files.getlist('files')
    if not files or len(files) == 0 or files[0].filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'}), 400
    
    # 保存所有上传的文件
    uploaded_files = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # 保存原始文件名，只做最小限度处理以避免路径注入
                filename = file.filename.replace('/', '_').replace('\\', '_')
                
                # 确保上传目录存在
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # 如果是PDF文件，保存到原始位置并转换为TXT
                if filename.lower().endswith('.pdf'):
                    # 先保存PDF文件到临时目录
                    pdf_dir = 'pdfs'
                    os.makedirs(pdf_dir, exist_ok=True)
                    pdf_path = os.path.join(pdf_dir, filename)
                    
                    logger.info(f"======= 开始保存PDF文件: {filename} =======")
                    file.save(pdf_path)
                    logger.info(f"======= PDF文件已保存: {pdf_path} =======")
                    
                    # 使用question1.py的函数提取文本并保存为TXT
                    try:
                        logger.info(f"======= 开始从PDF提取文本: {filename} =======")
                        extract_text_from_pdf(pdf_path)
                        
                        # 生成TXT文件名 - 与PDF文件同名但扩展名不同
                        txt_filename = f"{os.path.splitext(filename)[0]}.txt"
                        logger.info(f"======= 文本提取成功，保存为TXT: {txt_filename} =======")
                        add_log(f"PDF文件已转换为TXT: {filename} -> {txt_filename}")
                        
                        # 更新filename变量为TXT文件名，以便后续处理
                        filename = txt_filename
                    except Exception as e:
                        logger.error(f"======= 从PDF提取文本失败: {str(e)} =======")
                        add_log(f"PDF文本提取失败: {str(e)}", "ERROR")
                        errors.append({'filename': file.filename, 'error': f'PDF文本提取失败: {str(e)}'})
                        continue
                else:
                    # 非PDF文件直接保存到上传目录
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                
                add_log(f"文件已上传: {filename}")
                
                # 如果索引管理器存在，处理新文件
                if index_manager and is_monitoring:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    success = index_manager.process_new_file(file_path)
                    if success:
                        add_log(f"文件已添加到索引: {filename}")
                    else:
                        add_log(f"文件未能添加到索引: {filename}", "WARNING")
                
                uploaded_files.append(filename)
            except Exception as e:
                add_log(f"文件上传失败: {str(e)}", "ERROR")
                errors.append({'filename': file.filename, 'error': str(e)})
        else:
            errors.append({'filename': file.filename, 'error': '不支持的文件类型'})
    
    # 返回结果
    if len(uploaded_files) > 0:
        result = {
            'success': True,
            'uploaded_files': uploaded_files
        }
        if errors:
            result['partial_errors'] = errors
        return jsonify(result)
    else:
        return jsonify({'success': False, 'errors': errors}), 400

@app.route('/api/system_overview', methods=['GET'])
def api_system_overview():
    """获取系统概览数据API"""
    try:
        # 获取文件总数
        total_files = 0
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            total_files = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                          if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f)) and 
                          (f.endswith('.txt') or f.endswith('.pdf'))])
        
        # 获取已索引文件数量
        indexed_files = 0
        if os.path.exists(app.config['KB_FOLDER']) and os.path.exists(os.path.join(app.config['KB_FOLDER'], 'config.json')):
            try:
                with open(os.path.join(app.config['KB_FOLDER'], 'config.json'), 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    indexed_files = config.get('num_txt_files', 0) + config.get('num_excel_files', 0)
            except:
                pass
        
        # 获取问答记录数量
        qa_records = 0
        qa_history_file = 'qa_history.json'
        if os.path.exists(qa_history_file):
            try:
                with open(qa_history_file, 'r', encoding='utf-8') as f:
                    qa_history = json.load(f)
                    qa_records = len(qa_history)
            except:
                pass
        
        return jsonify({
            'total_files': total_files,
            'indexed_files': indexed_files,
            'qa_records': qa_records
        })
    except Exception as e:
        return jsonify({
            'total_files': 0,
            'indexed_files': 0,
            'qa_records': 0,
            'error': str(e)
        })

@app.route('/api/system_status', methods=['GET'])
def api_system_status():
    """获取系统详细状态API"""
    try:
        # 检查检索系统是否就绪
        retrieval_ready = os.path.exists(os.path.join(app.config['KB_FOLDER'], 'faiss_index.bin'))
        
        # 获取向量模型名称
        vector_model = 'all-MiniLM-L6-v2'  # 默认值
        if os.path.exists(os.path.join(app.config['KB_FOLDER'], 'config.json')):
            try:
                with open(os.path.join(app.config['KB_FOLDER'], 'config.json'), 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    vector_model = config.get('vector_model', vector_model)
            except:
                pass
        
        # 获取索引路径
        index_path = os.path.abspath(app.config['KB_FOLDER'])
        
        # 获取索引更新时间
        index_updated_time = None
        if os.path.exists(os.path.join(app.config['KB_FOLDER'], 'faiss_index.bin')):
            timestamp = os.path.getmtime(os.path.join(app.config['KB_FOLDER'], 'faiss_index.bin'))
            index_updated_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取系统启动时间
        system_start_time = system_status.get('start_time')
        
        return jsonify({
            'retrieval_ready': retrieval_ready,
            'vector_model': vector_model,
            'index_path': index_path,
            'index_updated_time': index_updated_time,
            'system_start_time': system_start_time,
            'is_monitoring': system_status.get('is_monitoring', False)
        })
    except Exception as e:
        return jsonify({
            'retrieval_ready': False,
            'vector_model': 'unknown',
            'index_path': app.config['KB_FOLDER'],
            'error': str(e)
        })

@app.route('/api/file_type_distribution', methods=['GET'])
def api_file_type_distribution():
    """获取文件类型分布数据API"""
    try:
        # 初始化文件类型计数
        file_types = {}
        
        # 检查上传目录是否存在
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            # 遍历目录中的文件
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path):
                    # 获取文件扩展名
                    ext = os.path.splitext(filename)[1].lower()
                    if not ext:
                        continue
                    
                    # 去除扩展名前的点
                    ext = ext[1:] if ext.startswith('.') else ext
                    
                    # 更友好的文件类型显示
                    if ext == 'pdf':
                        file_type = 'PDF'
                    elif ext == 'txt':
                        file_type = '文本文件'
                    elif ext in ['docx', 'doc']:
                        file_type = 'Word文档'
                    elif ext in ['xlsx', 'xls']:
                        file_type = 'Excel表格'
                    elif ext == 'md':
                        file_type = 'Markdown'
                    else:
                        file_type = ext.upper()
                    
                    # 统计文件类型数量
                    if file_type in file_types:
                        file_types[file_type] += 1
                    else:
                        file_types[file_type] = 1
        
        # 转换为前端期望的格式
        result = []
        for file_type, count in file_types.items():
            result.append({
                'file_type': file_type,
                'count': count
            })
        
        return jsonify({'data': result})
    except Exception as e:
        return jsonify({'data': [], 'error': str(e)})

@app.route('/api/process_unindexed_files', methods=['POST'])
def api_process_unindexed_files():
    """处理未索引文件API"""
    if not index_manager:
        return jsonify({'success': False, 'error': '索引管理器未初始化'})
    
    try:
        add_log("开始处理未索引文件...")
        
        # 获取上传文件夹中的所有文件
        upload_folder = app.config['UPLOAD_FOLDER']
        all_files = []
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                all_files.append(filename)
        
        # 筛选出未索引的文件
        unindexed_files = []
        for filename in all_files:
            if not index_manager.check_file_indexed(filename):
                unindexed_files.append(filename)
        
        if not unindexed_files:
            add_log("没有找到未索引的文件")
            return jsonify({
                'success': True, 
                'message': '没有找到未索引的文件',
                'total_processed': 0
            })
        
        # 处理未索引文件
        processed_files = []
        failed_files = []
        
        for filename in unindexed_files:
            file_path = os.path.join(upload_folder, filename)
            try:
                add_log(f"处理文件: {filename}")
                # 提取文本并索引文件
                success = index_manager.index_file(file_path)
                
                if success:
                    processed_files.append(filename)
                    add_log(f"成功索引文件: {filename}")
                else:
                    failed_files.append(filename)
                    add_log(f"索引文件失败: {filename}", "ERROR")
            except Exception as e:
                failed_files.append(filename)
                add_log(f"处理文件出错 {filename}: {str(e)}", "ERROR")
        
        # 返回处理结果
        return jsonify({
            'success': True,
            'total_processed': len(processed_files),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'message': f'成功处理 {len(processed_files)} 个文件，失败 {len(failed_files)} 个文件'
        })
        
    except Exception as e:
        error_msg = f"处理未索引文件时出错: {str(e)}"
        add_log(error_msg, "ERROR")
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/recent_files', methods=['GET'])
def api_recent_files():
    """获取最近上传的文件API"""
    try:
        files = []
        
        # 检查上传目录是否存在
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            # 获取目录中的所有文件
            file_list = []
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.pdf')):
                    # 获取文件修改时间
                    modified_time = os.path.getmtime(file_path)
                    # 获取文件大小
                    size = os.path.getsize(file_path)
                    
                    file_list.append({
                        'name': filename,
                        'path': file_path,
                        'uploaded_at': datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S"),
                        'size': size,
                        'indexed': True  # 简化处理，假设所有文件都已索引
                    })
            
            # 按修改时间排序，最新的在前
            file_list.sort(key=lambda x: x['uploaded_at'], reverse=True)
            
            # 只返回前5个文件
            files = file_list[:5]
        
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'files': [], 'error': str(e)})

@app.route('/api/document_content/<path:filename>', methods=['GET'])
def api_document_content(filename):
    """获取文档内容API"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        
        # 读取文件内容
        content = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            content = "无法读取文件内容"
        
        return jsonify({
            'success': True,
            'content': content,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'获取文档内容失败: {str(e)}'}), 500

@app.route('/api/chat_history', methods=['GET'])
def api_chat_history():
    """获取聊天历史记录API"""
    try:
        history = []
        qa_history_file = 'qa_history.json'
        
        if os.path.exists(qa_history_file):
            try:
                with open(qa_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass
        
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'history': [], 'error': str(e)})

@app.route('/api/save_chat', methods=['POST'])
def api_save_chat():
    """保存聊天记录API"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': '无效的数据'}), 400
        
        # 验证必要字段
        if 'question' not in data or 'answer' not in data:
            return jsonify({'success': False, 'error': '缺少必要字段'}), 400
        
        # 读取现有历史记录
        qa_history_file = 'qa_history.json'
        history = []
        
        if os.path.exists(qa_history_file):
            try:
                with open(qa_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass
        
        # 实例化HistoryManager，用于提取关键点
        history_manager = HistoryManager()
        # 提取关键点
        retrieved_docs = data.get('relevant_documents', [])
        key_points = history_manager._extract_key_points(data['answer'], retrieved_docs)
        
        # 添加新记录，包含关键点
        new_record = {
            'id': f"C{len(history) + 1:03d}",
            'query': data['question'],
            'answer': data['answer'],
            'key_points': key_points,  # 添加关键点字段
            'retrieved_docs': retrieved_docs
        }
        history.append(new_record)
        
        # 保存回文件
        with open(qa_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_chat_history', methods=['POST'])
def api_clear_chat_history():
    """清空聊天历史记录API"""
    try:
        qa_history_file = 'qa_history.json'
        
        # 清空历史记录
        with open(qa_history_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_history', methods=['GET'])
def api_export_history():
    """导出历史记录API"""
    try:
        format_type = request.args.get('format', 'excel')
        
        # 读取历史记录
        qa_history_file = 'qa_history.json'
        history = []
        
        if os.path.exists(qa_history_file):
            try:
                with open(qa_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass
        
        if not history:
            return jsonify({'success': False, 'error': '没有历史记录可导出'}), 404
        
        # 根据格式导出
        if format_type == 'excel':
            try:
                import pandas as pd
            except ImportError:
                return jsonify({'success': False, 'error': '缺少pandas库，无法导出Excel'}), 500
            
            # 创建一个临时Excel文件
            filename = f"qa_history_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
            
            # 准备数据
            data = []
            for record in history:
                data.append({
                    '问题编号': record.get('id', ''),
                    '问题': record.get('query', ''),
                    '关键点': record.get('key_points', ''),  # 添加关键点字段
                    '回答': record.get('answer', '')
                })
            
            # 创建DataFrame并保存为Excel
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False)
            
            # 返回文件下载
            return send_from_directory(os.getcwd(), filename, as_attachment=True)
        
        elif format_type == 'markdown':
            # 创建一个临时Markdown文件
            filename = f"qa_history_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
            
            # 生成Markdown内容
            content = "# 问答历史记录\n\n"
            for idx, record in enumerate(history, 1):
                content += f"## {idx}. {record.get('query', '')}\n\n"
                content += f"**时间**: {record.get('timestamp', '')}\n\n"
                if record.get('key_points'):
                    content += f"**关键点**:\n\n{record.get('key_points', '')}\n\n"
                content += f"**回答**:\n\n{record.get('answer', '')}\n\n"
                content += "---\n\n"
            
            # 保存文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 返回文件下载
            return send_from_directory(os.getcwd(), filename, as_attachment=True)
        
        elif format_type == 'txt':
            # 创建一个临时文本文件
            filename = f"qa_history_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            
            # 生成文本内容
            content = "问答历史记录\n\n"
            for idx, record in enumerate(history, 1):
                content += f"{idx}. 问题: {record.get('query', '')}\n"
                content += f"   时间: {record.get('timestamp', '')}\n"
                content += f"   回答: {record.get('answer', '')}\n\n"
            
            # 保存文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 返回文件下载
            return send_from_directory(os.getcwd(), filename, as_attachment=True)
        
        else:
            return jsonify({'success': False, 'error': '不支持的格式类型'}), 400
        
    except Exception as e:
        logger.error(f"导出历史记录API出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 启动应用
if __name__ == '__main__':
    # 确保上传目录和知识库目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['KB_FOLDER'], exist_ok=True)
    
    add_log("Web应用启动")
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000) 