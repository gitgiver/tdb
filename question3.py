#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件监控与索引增量更新服务
监控txts目录中文件变动，自动更新知识库索引
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# 导入question2.py中的必要组件
# 这种做法假设question2.py和question3.py在同一目录下
try:
    from question2 import DocumentProcessingPipeline, RAGSystem, logger
except ImportError:
    print("无法导入question2.py中的必要组件，请确保两个文件在同一目录下")
    sys.exit(1)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("file_monitor.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("FileMonitor")

class FileChangeTracker:
    """
    文件变更跟踪器
    记录已处理文件的元数据，用于检测文件内容变化
    """
    
    def __init__(self, tracker_file: str = "file_tracker.json"):
        """
        初始化文件变更跟踪器
        
        Args:
            tracker_file: 保存文件元数据的JSON文件路径
        """
        self.tracker_file = tracker_file
        self.file_metadata = self._load_tracker()
        
    def _load_tracker(self) -> Dict[str, Dict[str, Any]]:
        """从文件加载跟踪记录"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载文件跟踪记录失败: {str(e)}")
                return {}
        return {}
    
    def _save_tracker(self) -> None:
        """保存跟踪记录到文件"""
        try:
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"文件跟踪记录已保存到: {self.tracker_file}")
        except Exception as e:
            logger.error(f"保存文件跟踪记录失败: {str(e)}")
    
    def add_file(self, file_path: str) -> None:
        """
        添加文件到跟踪记录
        
        Args:
            file_path: 文件路径
        """
        try:
            if os.path.exists(file_path):
                stats = os.stat(file_path)
                self.file_metadata[file_path] = {
                    "size": stats.st_size,
                    "mtime": stats.st_mtime,
                    "last_processed": time.time()
                }
                self._save_tracker()
                logger.info(f"文件 {file_path} 已添加到跟踪记录")
            else:
                logger.warning(f"要添加的文件不存在: {file_path}")
        except Exception as e:
            logger.error(f"添加文件到跟踪记录失败: {file_path}, 错误: {str(e)}")
    
    def remove_file(self, file_path: str) -> None:
        """
        从跟踪记录中移除文件
        
        Args:
            file_path: 文件路径
        """
        if file_path in self.file_metadata:
            del self.file_metadata[file_path]
            self._save_tracker()
            logger.info(f"文件 {file_path} 已从跟踪记录中移除")
    
    def has_changed(self, file_path: str) -> bool:
        """
        检查文件是否已更改
        
        Args:
            file_path: 文件路径
            
        Returns:
            如果文件已更改，则返回True
        """
        if file_path not in self.file_metadata:
            # 新文件，需要处理
            return True
            
        try:
            if os.path.exists(file_path):
                stats = os.stat(file_path)
                old_meta = self.file_metadata[file_path]
                
                # 检查文件大小或修改时间是否变化,从而判断文件是否被修改过
                if stats.st_size != old_meta["size"] or stats.st_mtime > old_meta["mtime"]:
                    return True
            else:
                # 文件已删除，也认为是变化
                return True
        except Exception as e:
            logger.error(f"检查文件变化时出错: {file_path}, 错误: {str(e)}")
            # 发生错误时，谨慎起见将其视为已更改
            return True
            
        return False
    
    def get_all_tracked_files(self) -> Set[str]:
        """
        获取所有被跟踪的文件路径
        
        Returns:
            文件路径集合
        """
        return set(self.file_metadata.keys())


class IndexUpdateManager:
    """
    索引更新管理器
    处理知识库索引的增量更新和重建
    """
    
    def __init__(self, 
                knowledge_base_path: str = "knowledge_base",
                model_name: str = "all-MiniLM-L6-v2",
                chunk_size: int = 800,
                chunk_overlap: int = 200,
                api_key: str = "c66be1dbcd07484fa81efde6d883e410.O2y7leHySzSP7Ygc"):
        """
        初始化索引更新管理器
        
        Args:
            knowledge_base_path: 知识库路径
            model_name: 向量模型名称
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
            api_key: 智谱API密钥
        """
        self.knowledge_base_path = knowledge_base_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key
        
        # 记录已处理的文件和状态
        self.file_tracker = FileChangeTracker()
        
        # 初始化加载知识库，如果不存在则创建
        self._init_knowledge_base()
        
    def _init_knowledge_base(self) -> None:
        """初始化或加载现有知识库"""
        try:
            if os.path.exists(self.knowledge_base_path):
                logger.info(f"加载现有知识库: {self.knowledge_base_path}")
                self.pipeline = DocumentProcessingPipeline.load(
                    self.knowledge_base_path, 
                    self.model_name
                )
            else:
                logger.info(f"创建新知识库: {self.knowledge_base_path}")
                self.pipeline = DocumentProcessingPipeline(
                    data_dir="",  # 不需要处理txt文件
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    model_name=self.model_name,
                    output_dir=self.knowledge_base_path
                )
        except Exception as e:
            logger.error(f"初始化知识库失败: {str(e)}")
            raise
    
    def process_new_file(self, file_path: str) -> bool:
        """
        处理新增文件，更新知识库索引
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理成功返回True，否则返回False
        """
        try:
            logger.info(f"处理新增文件: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return False
            
            # 对于txt文件的特殊处理，避免使用PDF提取方法
            if file_path.lower().endswith('.txt'):
                try:
                    # 直接读取文本文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # 如果文本内容为空，记录警告
                    if not text_content.strip():
                        logger.warning(f"文本文件内容为空: {file_path}")
                    
                    # 手动创建文档对象，模拟DocumentProcessingPipeline.process_file的返回
                    # 假设pipeline.retrieval_system已经存在
                    if hasattr(self.pipeline, 'retrieval_system') and text_content.strip():
                        # 创建基本文档元数据
                        metadata = {
                            "source": file_path,
                            "file_type": "txt",
                            "created_at": time.time()
                        }
                        
                        # 使用pipeline的splitter分割文本
                        if hasattr(self.pipeline, 'splitter'):
                            doc = {"text": text_content, "metadata": metadata}
                            chunks = self.pipeline.splitter.split(doc)
                            
                            # 将分割后的文档添加到索引
                            if chunks and hasattr(self.pipeline.retrieval_system, 'build_index'):
                                self.pipeline.retrieval_system.build_index(chunks)
                                logger.info(f"文本文件内容已分割为 {len(chunks)} 个块并索引")
                                
                                # 更新跟踪记录
                                self.file_tracker.add_file(file_path)
                                
                                logger.info(f"文件 {file_path} 成功添加到索引，生成 {len(chunks)} 个文本块")
                                return True
                
                except Exception as inner_e:
                    logger.error(f"处理文本文件时出错: {file_path}, 错误: {str(inner_e)}")
                    # 继续使用原始处理方法作为备选
            
            # 原始处理逻辑保持不变，作为备选
            chunks = self.pipeline.process_file(file_path)
            
            # 更新跟踪记录
            self.file_tracker.add_file(file_path)
            
            logger.info(f"文件 {file_path} 成功添加到索引，生成 {len(chunks)} 个文本块")
            return True
        except Exception as e:
            logger.error(f"处理新增文件失败: {file_path}, 错误: {str(e)}")
            
            # 虽然处理失败，但仍将文件添加到跟踪记录
            # 这样避免反复尝试处理同一个有问题的文件
            self.file_tracker.add_file(file_path)
            
            return False
    
    def process_deleted_file(self, file_path: str) -> bool:
        """
        处理删除的文件，更新知识库索引
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理成功返回True，否则返回False
        """
        try:
            logger.info(f"处理删除文件: {file_path}")
            
            # 从跟踪记录中移除文件
            self.file_tracker.remove_file(file_path)
            logger.info(f"文件 {file_path} 已从跟踪记录中移除")
            
            # 获取监控目录
            monitored_dir = os.path.dirname(file_path)
            if not monitored_dir or monitored_dir == '.':
                monitored_dir = "txts"  # 默认目录
            
            # 重建索引以确保删除的文件内容从索引中移除
            logger.info(f"由于文件删除，开始重建索引...")
            success = self.rebuild_index(monitored_dir)
            if success:
                logger.info(f"索引重建成功，已完全移除文件 {file_path} 的内容")
            else:
                logger.warning(f"索引重建失败，可能无法完全移除文件 {file_path} 的内容")
            
            return success
        except Exception as e:
            logger.error(f"处理删除文件失败: {file_path}, 错误: {str(e)}")
            return False
    
    def process_modified_file(self, file_path: str) -> bool:
        """
        处理修改的文件，更新知识库索引
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理成功返回True，否则返回False
        """
        try:
            logger.info(f"处理修改文件: {file_path}")
            
            # 从跟踪记录中移除文件
            self.file_tracker.remove_file(file_path)
            logger.info(f"文件 {file_path} 已从跟踪记录中移除")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.warning(f"要更新的文件不存在: {file_path}")
                return False
            
            # 获取监控目录
            monitored_dir = os.path.dirname(file_path)
            if not monitored_dir or monitored_dir == '.':
                monitored_dir = "txts"  # 默认目录
            
            # 重建索引以确保修改的文件内容正确更新
            logger.info(f"由于文件修改，开始重建索引...")
            success = self.rebuild_index(monitored_dir)
            if success:
                logger.info(f"索引重建成功，已更新文件 {file_path} 的内容")
            else:
                logger.warning(f"索引重建失败，可能无法正确更新文件 {file_path} 的内容")
            
            return success
        except Exception as e:
            logger.error(f"处理修改文件失败: {file_path}, 错误: {str(e)}")
            return False
    
    def rebuild_index(self, data_dir: str = "txts") -> bool:
        """
        重建整个索引
        
        Args:
            data_dir: 数据目录
            
        Returns:
            重建成功返回True，否则返回False
        """
        try:
            logger.info(f"开始重建索引，数据目录: {data_dir}")
            
            # 重新初始化pipeline
            self.pipeline = DocumentProcessingPipeline(
                data_dir=data_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                model_name=self.model_name,
                output_dir=self.knowledge_base_path
            )
            
            # 处理所有文档
            success = self.pipeline.process_all_documents()
            
            if success:
                logger.info(f"索引重建成功，知识库路径: {self.knowledge_base_path}")
                
                # 更新文件跟踪记录
                self._update_file_tracker(data_dir)
                
                return True
            else:
                logger.error("索引重建失败")
                return False
        except Exception as e:
            logger.error(f"重建索引失败: {str(e)}")
            return False
    
    def _update_file_tracker(self, data_dir: str) -> None:
        """
        更新文件跟踪记录，添加目录中的所有文件
        
        Args:
            data_dir: 数据目录
        """
        try:
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.txt', '.pdf')):
                        file_path = os.path.join(root, file)
                        self.file_tracker.add_file(file_path)
            logger.info(f"文件跟踪记录已更新，当前跟踪 {len(self.file_tracker.get_all_tracked_files())} 个文件")
        except Exception as e:
            logger.error(f"更新文件跟踪记录失败: {str(e)}")
    
    def check_rebuild_needed(self, changed_files_count: int, total_files: int, threshold: float = 0.2) -> bool:
        """
        检查是否需要重建索引
        
        Args:
            changed_files_count: 已更改文件数量
            total_files: 总文件数量
            threshold: 阈值，更改文件比例超过此值时重建索引
            
        Returns:
            如果需要重建索引，则返回True
        """
        if total_files == 0:
            return False
            
        change_ratio = changed_files_count / total_files
        logger.info(f"文件变更比例: {change_ratio:.2f} ({changed_files_count}/{total_files})")
        
        return change_ratio > threshold


class DirectoryMonitorHandler(FileSystemEventHandler):
    """
    目录监控处理器
    处理文件系统事件
    """
    
    def __init__(self, index_manager: IndexUpdateManager, monitored_dir: str = "txts"):
        """
        初始化目录监控处理器
        
        Args:
            index_manager: 索引更新管理器
            monitored_dir: 被监控的目录
        """
        self.index_manager = index_manager
        self.monitored_dir = monitored_dir
        self.pending_changes = {}  # 暂存尚未处理的变更
        self.last_rebuild_time = 0  # 上次重建索引的时间
        
        # 防抖动间隔（秒）- 文件系统事件可能在短时间内触发多次
        self.debounce_interval = 3
    
    def on_created(self, event: FileSystemEvent) -> None:
        """
        文件创建事件处理
        
        Args:
            event: 文件系统事件
        """
        if event.is_directory or not self._should_process_file(event.src_path):
            return
            
        logger.info(f"检测到文件创建: {event.src_path}")
        self._queue_change(event.src_path, "created")
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """
        文件删除事件处理
        
        Args:
            event: 文件系统事件
        """
        if event.is_directory or not self._should_process_file(event.src_path):
            return
            
        logger.info(f"检测到文件删除: {event.src_path}")
        self._queue_change(event.src_path, "deleted")
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """
        文件修改事件处理
        
        Args:
            event: 文件系统事件
        """
        if event.is_directory or not self._should_process_file(event.src_path):
            return
            
        logger.info(f"检测到文件修改: {event.src_path}")
        self._queue_change(event.src_path, "modified")
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """
        文件移动事件处理
        
        Args:
            event: 文件系统事件
        """
        if event.is_directory:
            return
            
        # 处理源文件
        if self._should_process_file(event.src_path):
            logger.info(f"检测到文件移出: {event.src_path}")
            self._queue_change(event.src_path, "deleted")
            
        # 处理目标文件
        if self._should_process_file(event.dest_path):
            logger.info(f"检测到文件移入: {event.dest_path}")
            self._queue_change(event.dest_path, "created")
    
    def _should_process_file(self, file_path: str) -> bool:
        """
        检查是否应处理此文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            如果应处理此文件，则返回True
        """
        # 仅处理txt文本文件和pdf文件
        return file_path.endswith(('.txt', '.pdf')) and self.monitored_dir in file_path
    
    def _queue_change(self, file_path: str, change_type: str) -> None:
        """
        将变更添加到队列
        
        Args:
            file_path: 文件路径
            change_type: 变更类型 ("created", "deleted", "modified")
        """
        self.pending_changes[file_path] = {
            "type": change_type,
            "timestamp": time.time()
        }
    
    def process_pending_changes(self) -> None:
        """处理挂起的变更"""
        current_time = time.time()
        
        # 收集需要处理的变更
        changes_to_process = {}
        for file_path, change in list(self.pending_changes.items()):
            # 仅处理超过防抖动间隔的变更
            if current_time - change["timestamp"] > self.debounce_interval:
                changes_to_process[file_path] = change
                del self.pending_changes[file_path]
        
        if not changes_to_process:
            return
            
        # 分类变更
        created_files = []
        deleted_files = []
        modified_files = []
        
        for file_path, change in changes_to_process.items():
            if change["type"] == "created":
                created_files.append(file_path)
            elif change["type"] == "deleted":
                deleted_files.append(file_path)
            elif change["type"] == "modified":
                modified_files.append(file_path)
        
        # 获取跟踪的文件总数
        total_tracked_files = len(self.index_manager.file_tracker.get_all_tracked_files())
        total_changed_files = len(created_files) + len(deleted_files) + len(modified_files)
        
        # 检查是否需要重建索引
        if (total_changed_files > 0 and 
            self.index_manager.check_rebuild_needed(total_changed_files, total_tracked_files) and
            current_time - self.last_rebuild_time > 300):  # 至少5分钟间隔
            
            logger.info(f"变更文件过多 ({total_changed_files}/{total_tracked_files})，触发索引重建")
            self.index_manager.rebuild_index(self.monitored_dir)
            self.last_rebuild_time = current_time
            return
        
        # 增量处理变更
        logger.info(f"开始处理 {len(changes_to_process)} 个文件变更")
        
        # 处理删除的文件
        for file_path in deleted_files:
            self.index_manager.process_deleted_file(file_path)
        
        # 处理修改的文件
        for file_path in modified_files:
            self.index_manager.process_modified_file(file_path)
        
        # 处理新增的文件
        for file_path in created_files:
            self.index_manager.process_new_file(file_path)
        
        logger.info(f"已处理 {len(changes_to_process)} 个文件变更")


def main():
    """主函数"""
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="文件监控和索引增量更新服务")
    
    # 基础参数
    parser.add_argument("--data_dir", default="txts", help="要监控的文件目录")
    parser.add_argument("--knowledge_base", default="knowledge_base", help="知识库目录")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2", help="向量化模型名称")
    parser.add_argument("--chunk_size", type=int, default=800, help="分块大小")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="分块重叠大小")
    parser.add_argument("--api_key", default="c66be1dbcd07484fa81efde6d883e410.O2y7leHySzSP7Ygc", help="智谱API密钥")
    parser.add_argument("--initial_scan", action="store_true", help="启动时扫描并处理目录中的所有文件")
    
    args = parser.parse_args()
    
    # 确保监控目录存在
    if not os.path.exists(args.data_dir):
        logger.info(f"创建监控目录: {args.data_dir}")
        os.makedirs(args.data_dir)
    
    # 初始化索引更新管理器
    index_manager = IndexUpdateManager(
        knowledge_base_path=args.knowledge_base,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        api_key=args.api_key
    )
    
    # 如果请求初始扫描，则处理目录中的所有文件
    if args.initial_scan:
        logger.info(f"执行初始扫描: {args.data_dir}")
        index_manager.rebuild_index(args.data_dir)
    
    # 初始化目录监控处理器
    event_handler = DirectoryMonitorHandler(index_manager, args.data_dir)
    
    # 设置观察者
    observer = Observer()
    observer.schedule(event_handler, args.data_dir, recursive=True)
    observer.start()
    
    logger.info(f"开始监控目录: {args.data_dir}")
    
    try:
        while True:
            # 处理挂起的变更
            event_handler.process_pending_changes()
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("文件监控服务已停止")
    
    observer.join()


if __name__ == "__main__":
    main() 