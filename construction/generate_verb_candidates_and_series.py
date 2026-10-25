"""
Google Ngram Timeseries 工具 - 用于获取和分析词语使用频率数据
改进的模块化代码结构，便于维护和扩展
"""
import requests
import json
import matplotlib.pyplot as plt
import urllib.parse
import os
import matplotlib.ticker as mticker
import time
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union, Set

# ========== 异常类 ==========
class NGramException(Exception):
    """NGram API相关异常的基类"""
    pass

class APIAccessError(NGramException):
    """API访问异常，表示由于网络或API访问问题导致数据获取失败"""
    pass

class DataProcessingError(NGramException):
    """数据处理错误，表示在处理数据时发生错误"""
    pass

# ========== 数据模型 ==========
@dataclass
class NGramDataPoint:
    """单个Ngram数据点"""
    ngram: str
    timeseries: List[float]
    parent_ngram: Optional[str] = None
    type: str = "UNKNOWN"  # 可以是 "SPELLING" 或 "MISSPELLING" 或其他

@dataclass
class NGramResult:
    """Ngram查询结果"""
    data_points: List[NGramDataPoint] = field(default_factory=list)
    years_axis: List[int] = field(default_factory=list)
    
    def has_valid_data(self) -> bool:
        """检查是否包含有效数据"""
        return len(self.data_points) > 0 and len(self.years_axis) > 0
    
    def to_dict(self) -> Dict:
        """将结果转换为可序列化的字典"""
        return {
            "years_axis": self.years_axis,
            "ngrams_data": {
                dp.ngram: {
                    "ngram": dp.ngram,
                    "parent_ngram": dp.parent_ngram,
                    "timeseries": dp.timeseries,
                    "type": dp.type
                } for dp in self.data_points
            }
        }

@dataclass
class ProcessingStatus:
    """处理状态跟踪"""
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    
    def increment_processed(self, success: bool = True) -> None:
        """增加已处理计数"""
        self.processed_items += 1
        if success:
            self.successful_items += 1
        else:
            self.failed_items += 1
    
    def get_progress_percentage(self) -> float:
        """获取进度百分比"""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

# ========== API服务 ==========
class NGramAPIService:
    """
    Google Ngram API服务
    负责与API通信并获取Ngram数据
    """
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: int = 5, 
                 retry_delay_503: int = 30,
                 api_call_delay: int = 3):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_503 = retry_delay_503
        self.api_call_delay = api_call_delay
    
    def fetch_ngram_data(self, 
                        phrases: List[str], 
                        start_year: int = 1800, 
                        end_year: int = 2022, 
                        corpus: str = 'english', 
                        smoothing: int = 3) -> Tuple[List[Dict], Optional[List[int]]]:
        """
        获取Google Ngram数据，包含重试机制
        
        :return: 包含每个词组频率数据和年份信息的元组 (data, years_axis)
        """
        query = ','.join(phrases)
        encoded_query = urllib.parse.quote(query)
        url = f"https://books.google.com/ngrams/json?content={encoded_query}&year_start={start_year}&year_end={end_year}&corpus={corpus}&smoothing={smoothing}"

        retries = 0
        while retries < self.max_retries:
            current_delay = self.retry_delay
            try:
                print(f"尝试获取: {url} (尝试 {retries + 1}/{self.max_retries})")
                response = requests.get(url, timeout=30) 
                response.raise_for_status()  
                data = response.json()
                
                if not data:
                    print(f"API未返回数据: {query} 在语料库: {corpus} (API返回空列表).")
                    return None, None 
                
                # 检查每个返回的词条是否真的包含时间序列
                all_items_have_timeseries = True
                for item in data:
                    if not item.get('timeseries'):
                        all_items_have_timeseries = False
                        print(f"未找到ngram '{item.get('ngram', 'N/A')}' 的时序数据: {query}")
                        break
                
                if not all_items_have_timeseries and not any(item.get('timeseries') for item in data):
                    print(f"查询: {query} 中没有任何ngram有有效的时序数据.")
                    return data, None
                
                # 确定年份轴
                years_axis = None
                for item in data:
                    if item.get('timeseries'):
                        num_years = len(item['timeseries'])
                        years_axis = list(range(start_year, start_year + num_years))
                        if len(years_axis) > (end_year - start_year + 1):
                            years_axis = list(range(start_year, end_year + 1))
                        break
                
                return data, years_axis
                
            except requests.exceptions.HTTPError as e:
                print(f"HTTP错误: {e}. URL: {url}")
                if e.response is not None and e.response.status_code == 503:
                    print(f"服务器返回503 (Service Unavailable). 将在 {self.retry_delay_503} 秒后重试...")
                    current_delay = self.retry_delay_503
                    retries += 1
                    if retries < self.max_retries:
                        time.sleep(current_delay)
                        continue
                    else:
                        print(f"因持续的HTTP 503错误，获取 {query} 失败 (已达最大重试次数).")
                        raise APIAccessError(f"在 {self.max_retries} 次重试后，仍因持续的HTTP 503错误无法获取 {query}. 最后错误: {e}")
                elif e.response is not None and 400 <= e.response.status_code < 500:
                    print(f"客户端错误 (4xx): {e.response.status_code}. 通常不可重试。")
                    return None, None
                else:
                    print(f"其他HTTP错误. {current_delay} 秒后重试...")
                    retries += 1
                    if retries < self.max_retries:
                        time.sleep(current_delay)
                        continue
                    else:
                        print(f"因持续的HTTP错误，获取 {query} 失败 (已达最大重试次数).")
                        raise APIAccessError(f"在 {self.max_retries} 次重试后，仍因持续的HTTP错误无法获取 {query}. 最后错误: {e}")
            
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                print(f"网络/连接错误: {e}. {current_delay} 秒后重试...")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(current_delay)
                else:
                    print(f"因持续的网络/连接错误，获取 {query} 失败 (已达最大重试次数).")
                    raise APIAccessError(f"在 {self.max_retries} 次重试后，仍因持续的网络/连接问题无法获取 {query}. 最后错误: {e}")
                    
            except requests.exceptions.RequestException as e:
                print(f"请求错误: {e}. {current_delay} 秒后重试...")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(current_delay)
                else:
                    print(f"因持续的请求错误，获取 {query} 失败 (已达最大重试次数).")
                    raise APIAccessError(f"在 {self.max_retries} 次重试后，仍因持续的请求问题无法获取 {query}. 最后错误: {e}")
                    
            except json.JSONDecodeError:
                print(f"无法解析JSON响应，URL可能无效或无数据: {url}")
                print(f"响应文本: {response.text if 'response' in locals() else 'N/A'}")
                return None, None
        
        return None, None
    
    def wait_between_calls(self) -> None:
        """在API调用之间等待，避免过度请求"""
        time.sleep(self.api_call_delay)

# ========== 数据处理器 ==========
class DataProcessor(ABC):
    """数据处理器抽象基类，定义处理数据的接口"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据的抽象方法"""
        pass

class NGramDataProcessor(DataProcessor):
    """Ngram数据处理器，负责处理和组织API返回的数据"""
    
    def __init__(self, api_service: NGramAPIService):
        self.api_service = api_service
        
    def process(self, data: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """清理和处理输入数据字典"""
        if not data:
            return {}
            
        # 使用深拷贝以便修改和后续比较
        cleaned_data = copy.deepcopy(data)
        data_was_changed = False

        print(f"开始清洗输入数据...")
        for correct_spell, misspellings in cleaned_data.items():
            if not isinstance(misspellings, list):
                print(f"  警告: '{correct_spell}' 的错误拼写数据不是列表格式，跳过清洗此条目。")
                continue

            original_misspellings_count = len(misspellings)
            # 过滤掉与 correct_spell 完全相同的错误拼写（区分大小写）
            misspellings[:] = [m for m in misspellings if m != correct_spell]

            if len(misspellings) != original_misspellings_count:
                data_was_changed = True
                print(f"  对于 '{correct_spell}'，移除了 {original_misspellings_count - len(misspellings)} 个与其自身相同的错误拼写。")
        
        return cleaned_data
    
    def fetch_correct_spelling_data(self, correct_spell: str, start_year: int, end_year: int, 
                                corpus: str, smoothing: int) -> Tuple[Optional[NGramDataPoint], Optional[List[int]]]:
        """获取正确拼写的数据"""
        ngram_data, years_axis = self.api_service.fetch_ngram_data(
            [correct_spell], start_year, end_year, corpus, smoothing
        )
        self.api_service.wait_between_calls()
        
        if ngram_data and years_axis and ngram_data[0].get('timeseries'):
            return NGramDataPoint(
                ngram=ngram_data[0]['ngram'],
                timeseries=ngram_data[0]['timeseries'],
                parent_ngram=None,
                type="SPELLING"
            ), years_axis
        return None, None
    
    def fetch_misspellings_batch(self, misspellings: List[str], correct_spell: str,
                                start_year: int, end_year: int, corpus: str, smoothing: int) -> List[NGramDataPoint]:
        """获取一批错误拼写的数据"""
        result_points = []
        
        batch_ngram_data, _ = self.api_service.fetch_ngram_data(
            misspellings, start_year, end_year, corpus, smoothing
        )
        self.api_service.wait_between_calls()
        
        if batch_ngram_data:
            for item_data in batch_ngram_data:
                misspelling_name = item_data.get('ngram')
                if misspelling_name and item_data.get('timeseries'):
                    print(f"成功获取 '{misspelling_name}' 的数据.")
                    result_points.append(NGramDataPoint(
                        ngram=misspelling_name,
                        timeseries=item_data['timeseries'],
                        parent_ngram=correct_spell,
                        type="MISSPELLING"
                    ))
                elif misspelling_name:
                    print(f"未能获取 '{misspelling_name}' 的有效时序数据 (API返回无序列数据).")
        
        return result_points

# ========== 数据持久化 ==========
class DataPersistence:
    """数据持久化管理，处理文件读写操作"""
    
    @staticmethod
    def save_to_json(data: Any, filepath: str) -> None:
        """将数据保存为JSON文件"""
        if data:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"数据已保存到 {filepath}")
        else:
            print("没有数据可保存。")
    
    @staticmethod
    def load_from_json(filepath: str) -> Any:
        """从JSON文件加载数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误: 文件 {filepath} 未找到.")
            return None
        except json.JSONDecodeError:
            print(f"错误: 文件 {filepath} 不是有效的JSON格式.")
            return None


# ========== 处理协调器 ==========

class ProcessingCoordinator:
    """
    处理协调器
    负责协调整个处理流程，包括加载数据、处理数据、保存结果等
    """
    
    def __init__(self, 
                 api_service: NGramAPIService, 
                 data_processor: NGramDataProcessor,
                 data_persistence: DataPersistence,
                 batch_size: int = 5):
        self.api_service = api_service
        self.data_processor = data_processor
        self.data_persistence = data_persistence
        self.batch_size = batch_size
        self.status = ProcessingStatus()
        
    def process_data_set(self, 
                        input_filepath: str, 
                        clean_output_filepath: str, 
                        timeseries_output_filepath: str,
                        start_year: int = 1800, 
                        end_year: int = 2022, 
                        corpus: str = 'english', 
                        smoothing: int = 3) -> None:
        """处理数据集"""
        try:
            # 步骤1: 加载和清洗输入数据
            print(f"\n步骤 1: 正在加载并清洗输入文件: {input_filepath}")
            input_data = self.data_persistence.load_from_json(input_filepath)
            if not input_data:
                print(f"错误: 未能加载输入文件 {input_filepath}。处理终止。")
                return
                
            cleaned_input = self.data_processor.process(input_data)
            # 保存清洗后的输入数据
            self.data_persistence.save_to_json(cleaned_input, input_filepath)
            
            # 步骤2: 加载之前保存的处理进度
            print(f"\n步骤 2: 正在加载已保存的处理进度...")
            cleaned_output = self.data_persistence.load_from_json(clean_output_filepath) or {}
            timeseries_saved = self.data_persistence.load_from_json(timeseries_output_filepath)
            
            # 初始化结果存储
            ngram_result = NGramResult()
            if timeseries_saved and "ngrams_data" in timeseries_saved and "years_axis" in timeseries_saved:
                # 从之前保存的结果恢复
                ngram_result.years_axis = timeseries_saved["years_axis"] or []
                
                for ngram, data in timeseries_saved["ngrams_data"].items():
                    ngram_result.data_points.append(NGramDataPoint(
                        ngram=data["ngram"],
                        timeseries=data["timeseries"],
                        parent_ngram=data.get("parent_ngram"),
                        type=data.get("type", "UNKNOWN")
                    ))
            
            # 设置进度跟踪
            self.status.total_items = len(cleaned_input)
            self.status.processed_items = len(cleaned_output)
            
            self._process_items(
                cleaned_input, 
                cleaned_output, 
                ngram_result,
                start_year, 
                end_year, 
                corpus, 
                smoothing, 
                clean_output_filepath, 
                timeseries_output_filepath
            )
            
        except APIAccessError as e:
            print(f"\n错误: API访问问题导致处理中断: {e}")
            self._handle_interruption(
                cleaned_output, 
                ngram_result,
                clean_output_filepath, 
                timeseries_output_filepath
            )
        except KeyboardInterrupt:
            print("\n用户手动中断。正在保存当前进度...")
            self._handle_interruption(
                cleaned_output, 
                ngram_result,
                clean_output_filepath, 
                timeseries_output_filepath
            )
        except Exception as e:
            print(f"\n处理过程中发生意外错误: {e}")
            import traceback
            traceback.print_exc()
            self._handle_interruption(
                cleaned_output, 
                ngram_result,
                clean_output_filepath, 
                timeseries_output_filepath
            )
    
    def _process_items(self, 
                      cleaned_input: Dict[str, List[str]], 
                      cleaned_output: Dict[str, List[str]],
                      ngram_result: NGramResult,
                      start_year: int, 
                      end_year: int, 
                      corpus: str, 
                      smoothing: int,
                      clean_output_filepath: str,
                      timeseries_output_filepath: str) -> None:
        """处理各个项目"""
        print(f"总共需要处理 {self.status.total_items} 个正确拼写词条。")
        
        for correct_spell, misspellings in cleaned_input.items():
            # 检查是否已处理过
            if correct_spell in cleaned_output:
                print(f"正确拼写 '{correct_spell}' 已在之前的运行中处理过，跳过。")
                continue
                
            print(f"\n处理进度: 正确拼写 {self.status.processed_items + 1}/{self.status.total_items} ('{correct_spell}')")
            print(f"正在处理正确拼写: {correct_spell}")
            
            # 获取正确拼写数据
            correct_data_point, years_axis = self.data_processor.fetch_correct_spelling_data(
                correct_spell, start_year, end_year, corpus, smoothing
            )
            
            if correct_data_point and years_axis:
                # 添加到结果中
                ngram_result.data_points.append(correct_data_point)
                if not ngram_result.years_axis:
                    ngram_result.years_axis = years_axis
                print(f"成功获取 '{correct_spell}' 的数据.")
                
                # 处理错误拼写
                successful_misspellings = []
                if misspellings:
                    print(f"正在处理 '{correct_spell}' 的 {len(misspellings)} 个错误拼写...")
                    for i in range(0, len(misspellings), self.batch_size):
                        batch = misspellings[i:i+self.batch_size]
                        print(f"查询批次: {batch}")
                        
                        misspelling_points = self.data_processor.fetch_misspellings_batch(
                            batch, correct_spell, start_year, end_year, corpus, smoothing
                        )
                        
                        # 添加到结果
                        for point in misspelling_points:
                            ngram_result.data_points.append(point)
                            successful_misspellings.append(point.ngram)
                
                # 更新处理状态
                cleaned_output[correct_spell] = successful_misspellings
            else:
                print(f"未能获取 '{correct_spell}' 的有效时序数据。将记录为空并跳过其错误拼写处理。")
                # 记录为空列表表示已处理但未成功
                cleaned_output[correct_spell] = []
            
            # 更新进度并保存
            self.status.increment_processed(correct_data_point is not None)
            self._save_progress(cleaned_output, ngram_result, clean_output_filepath, timeseries_output_filepath)
            print(f"完成处理 '{correct_spell}' 及其错误拼写。当前有效错误拼写: {cleaned_output.get(correct_spell, [])}")
        
        print("\n所有数据处理完成。")
        print(f"清洗后的数据已保存到: {clean_output_filepath}")
        print(f"所有时间序列数据已保存到: {timeseries_output_filepath}")
    
    def _save_progress(self, 
                      cleaned_output: Dict[str, List[str]], 
                      ngram_result: NGramResult,
                      clean_output_filepath: str,
                      timeseries_output_filepath: str) -> None:
        """保存当前处理进度"""
        self.data_persistence.save_to_json(cleaned_output, clean_output_filepath)
        if not ngram_result.years_axis:
            # 确保至少有年份轴
            ngram_result.years_axis = list(range(1800, 2023))
        self.data_persistence.save_to_json(ngram_result.to_dict(), timeseries_output_filepath)
        print(f"已更新 {clean_output_filepath} 和 {timeseries_output_filepath}")
    
    def _handle_interruption(self, 
                           cleaned_output: Dict[str, List[str]], 
                           ngram_result: NGramResult,
                           clean_output_filepath: str,
                           timeseries_output_filepath: str) -> None:
        """处理中断情况，保存当前进度"""
        print("正在保存当前进度...")
        self._save_progress(cleaned_output, ngram_result, clean_output_filepath, timeseries_output_filepath)
        print(f"已处理 {self.status.processed_items}/{self.status.total_items} 个正确拼写。进度已保存。")


# ========== 主程序 ==========

def main():
    # 定义文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_json_path = os.path.join(base_dir, "DataSet", "longAdjs_llm.json")
    clean_output_json_path = os.path.join(base_dir, "DataSet", "longAdjs_llm_clean.json")
    timeseries_output_json_path = os.path.join(base_dir, "DataSet", "longAdjs_llm_timeseries.json")


    # 配置参数
    START_YEAR = 1800
    END_YEAR = 2022
    CORPUS = 'english' 
    SMOOTHING = 0
    API_CALL_DELAY = 3
    MAX_RETRIES = 4
    RETRY_DELAY = 10
    RETRY_DELAY_503 = 30
    BATCH_SIZE_MISSPELLINGS = 10
    
    # 初始化组件
    api_service = NGramAPIService(
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY,
        retry_delay_503=RETRY_DELAY_503,
        api_call_delay=API_CALL_DELAY
    )
    
    data_processor = NGramDataProcessor(api_service)
    data_persistence = DataPersistence()
    
    coordinator = ProcessingCoordinator(
        api_service=api_service,
        data_processor=data_processor,
        data_persistence=data_persistence,
        batch_size=BATCH_SIZE_MISSPELLINGS
    )
    
    # 执行处理
    coordinator.process_data_set(
        input_filepath=input_json_path,
        clean_output_filepath=clean_output_json_path,
        timeseries_output_filepath=timeseries_output_json_path,
        start_year=START_YEAR,
        end_year=END_YEAR,
        corpus=CORPUS,
        smoothing=SMOOTHING
    )

if __name__ == '__main__':
    main()