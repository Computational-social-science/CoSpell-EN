'''
之前有592个副词没有任何数据 
再用Qwen2.5-32B-Instruct-AWQ重新生成 生成的结果再去GBNC上查找 如果没有命中就最多重复三轮
'''

import re
import json
import torch
import os
import time
import copy
import requests
import urllib.parse
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义示例中的错误变体，用于检测重复
example_misspellings = {
    "beautiful": ["beutiful", "beautifull", "beatiful", "beautifl", "beautifel"],
    "touch": ["tuch", "touh", "touchh", "tuchh", "toch"],
    "computer": ["comuter", "compuer", "computor", "copmuter", "compyter"],
    "quickly": ["quikly", "quickley", "qwikly", "quckly", "quiclky"]
}

def load_model(model_path):
    """加载 LLaMA 模型和 tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        print(f"模型加载错误: {str(e)}")
        raise

def generate_misspellings(model, tokenizer, word, pos, temperature=0.8, top_p=0.9, max_new_tokens=50):
    """生成拼写错误"""
    prompt = f"""# English Word Misspelling Generator

## Task Description
Generate common misspellings of English words while maintaining their POS consistency.

## Output Format
Returns a JSON array with plausible typos.

## Rules
- The generated result is part-of-speech consistent with the original word
- Avoid duplication
- Exclude original words

## Examples
### Example 1: Adjective
Input: ```json
{{"word": "beautiful", "pos": "Adjective"}} 
``` 
Output:```json
["beutiful", "beautifull", "beatiful", "beautifl", "beautifel"]
```

### Example 2: Verb
Input: ```json
{{"word": "touch", "pos": "Verb"}} 
```
Output: ```json
["tuch", "touh", "touchh", "tuchh", "toch"]
```
### Example 3: Noun
Input: ```json
{{"word": "computer", "pos": "Noun"}}
```
Output:```json
["comuter", "compuer", "computor", "copmuter", "compyter"]
```

### Example 4: Adverb
Input: ```json
{{"word": "quickly", "pos": "Adverb"}} 
```
Output:```json
["quikly", "quickley", "qwikly", "quckly", "quiclky"]
```

## Current Task 
Input: ```json 
{{"word": "{word}", "pos": "{pos}"}}
``` 
Output:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成的原始文本:\n{generated_text}")  # 调试输出
    
    # 增强的结果提取
    try:
        # 匹配最后出现的JSON数组
        json_matches = re.findall(r'\[[\s\S]*?\]', generated_text)
        if json_matches:
            last_json = json_matches[-1]  # 获取最后一个匹配
            misspellings = json.loads(last_json)
            
            # 处理可能的字典格式输出
            processed_misspellings = []
            for m in misspellings:
                if isinstance(m, dict):
                    # 如果是字典格式，尝试提取'typo'字段
                    if 'typo' in m:
                        processed_misspellings.append(m['typo'])
                elif isinstance(m, str):
                    # 如果是字符串格式，直接添加
                    processed_misspellings.append(m)
            
            # 改进的重复检测逻辑
            is_example_duplicate = False
            for example_word, example_vars in example_misspellings.items():
                if example_word == word:
                    # 如果是同一个单词，允许部分重复
                    continue
                # 计算与示例变体的重复数量
                common_variants = set(processed_misspellings) & set(example_vars)
                if len(common_variants) >= 3:  # 只有大量重复才判定为复制示例
                    is_example_duplicate = True
                    print(f"警告: 生成结果与示例 {example_word} 高度重合")
                    break
            
            if is_example_duplicate:
                return []
                
            # 验证和清理结果
            valid_misspellings = [
                m.strip() for m in processed_misspellings 
                if isinstance(m, str) and m.strip() != word
            ]
            
            if valid_misspellings:
                return list(set(valid_misspellings))
            else:
                print("未找到有效的拼写错误变体")
        else:
            print("未找到JSON数组格式的输出")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"尝试解析的文本: {last_json if 'last_json' in locals() else 'None'}")
    except Exception as e:
        print(f"处理结果时出错: {str(e)}")
        print(f"原始JSON: {last_json if 'last_json' in locals() else 'None'}")
    
    return []

def try_generate_misspellings(model, tokenizer, word, pos, max_retries=2):
    """尝试生成拼写错误，如果失败则重试"""
    attempts = 0
    while attempts <= max_retries:
        if attempts > 0:
            print(f"第{attempts}次重试生成 {word}...")
            
        misspellings = generate_misspellings(model, tokenizer, word, pos)
        if misspellings:  # 如果成功生成
            return misspellings, attempts
            
        attempts += 1
    
    return [], attempts  # 所有尝试都失败

def load_test_words(json_path):
    """从JSON文件加载测试单词"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件错误: {str(e)}")
        raise

def check_in_google_books(word, max_retries=3, retry_delay=5):
    """
    通过Google Books Ngram API验证单词是否存在
    
    参数:
        word: 要验证的单词
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
    
    返回:
        bool: 如果单词存在于Google Books中返回True，否则返回False
    """
    encoded_word = urllib.parse.quote(word)
    url = f"https://books.google.com/ngrams/json?content={encoded_word}&year_start=1800&year_end=2022&corpus=english&smoothing=3"
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data and data[0].get('timeseries'):
                return True
            return False
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 503:
                print(f"服务器返回503 (Service Unavailable). 将在 {retry_delay} 秒后重试...")
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"因持续的HTTP 503错误，验证 {word} 失败 (已达最大重试次数).")
                    return False
            else:
                print(f"HTTP错误验证 {word}: {e}")
                return False
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"网络/连接错误验证 {word}: {e}. {retry_delay} 秒后重试...")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
            else:
                print(f"因持续的网络/连接错误，验证 {word} 失败 (已达最大重试次数).")
                return False
                
        except Exception as e:
            print(f"验证 {word} 时发生未知错误: {e}")
            return False
    
    return False

def save_results(results, output_path, invalid_output_path):
    """保存结果到JSON文件"""
    try:
        formatted_results = {}
        invalid_results = {}

        for word, variants in results.items():
            if variants:  # 如果模型生成过变体（不管是否都合法）
                valid_variants = []
                invalid_variants = []

                for variant in variants:
                    if check_in_google_books(variant):
                        valid_variants.append(variant)
                    else:
                        invalid_variants.append(variant)

                if valid_variants:
                    formatted_results[word] = valid_variants
                else:
                    print(f"{word} 所有变体都无效，跳过写入 formatted_results")

                if invalid_variants:
                    invalid_results[word] = invalid_variants
            else:
                print(f"{word} 没有生成任何变体，跳过写入")  # 这也是失败

        # 写入有效变体文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存到: {output_path}")

        # 写入无效变体文件（如果有）
        if invalid_results:
            with open(invalid_output_path, 'w', encoding='utf-8') as f_invalid:
                json.dump(invalid_results, f_invalid, indent=2, ensure_ascii=False)
            print(f"⚠️ 无效变体已保存到: {invalid_output_path}")

        return {
            "success": len(formatted_results),
            "failed": len(results) - len(formatted_results)
        }

    except Exception as e:
        print(f"保存结果错误: {str(e)}")
        raise


def load_checkpoint(output_path):
    """加载已有的处理结果"""
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"加载检查点错误: {str(e)}")
        return {}

def save_intermediate_results(results, output_path, backup=True):
    """保存中间结果，可选择创建备份"""
    try:
        if backup and os.path.exists(output_path):
            backup_path = f"{output_path}.bak"
            os.rename(output_path, backup_path)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"中间结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存中间结果错误: {str(e)}")

def process_all_words(model, tokenizer, test_words, output_path, save_interval=10, max_global_retries=3):
    """处理所有单词，包含断点续传和定期保存功能（只保留合法变体）"""
    from collections import defaultdict
    results = load_checkpoint(output_path)
    invalid_results = defaultdict(list)  # 可选：保存无效变体
    remaining_words = [item for item in test_words if item["word"] not in results or not results[item["word"]]]
    retry_count = 0
    words_processed = 0

    print(f"已加载{len(results)}个已处理结果，剩余{len(remaining_words)}个待处理")

    while remaining_words and retry_count < max_global_retries:
        failed_words = []

        for item in remaining_words:
            word = item["word"]
            pos = item["pos"]
            print(f"\n正在测试: {word} ({pos}) - 第{retry_count + 1}轮")

            misspellings, attempts = try_generate_misspellings(model, tokenizer, word, pos)

            if misspellings:
                # 验证哪些拼写在 Google Books 中合法
                valid_misspellings = [v for v in misspellings if check_in_google_books(v)]
                invalid_misspellings = [v for v in misspellings if v not in valid_misspellings]

                if valid_misspellings:
                    results[word] = valid_misspellings
                    print(f"✅ 成功生成: {word} -> {valid_misspellings}")
                else:
                    failed_words.append(item)
                    print(f"⚠️ 所有变体无效: {word} -> {misspellings}")
                
                if invalid_misspellings:
                    invalid_results[word].extend(invalid_misspellings)
            else:
                # 连原始 misspellings 都没有生成
                failed_words.append(item)
                print(f"⚠️ 暂未成功: {word}")

            words_processed += 1
            if words_processed % save_interval == 0:
                save_intermediate_results(results, output_path)
        
        if failed_words:
            print(f"\n=== 第{retry_count + 1}轮完成 ===")
            print(f"仍有 {len(failed_words)} 个单词待处理")
            remaining_words = failed_words
            retry_count += 1
            save_intermediate_results(results, output_path)
        else:
            break

    # 最后保存无效变体（可选）
    if invalid_results:
        invalid_path = output_path.replace(".json", "_invalid.json")
        with open(invalid_path, "w", encoding="utf-8") as f:
            json.dump(invalid_results, f, indent=2, ensure_ascii=False)
        print(f"⚠️ 无效变体已保存到: {invalid_path}")

    return results

if __name__ == "__main__":
    model_path = "/data1/css/models/Qwen2.5-32B-Instruct-AWQ"
    json_path = "/path/to/cospell-home/misspell/DataAnalysis/FrequencyEffect/Dataset/VERB/compare_verb_results.json"
    output_path = "/path/to/cospell-home/misspell/DataAnalysis/FrequencyEffect/Dataset/VERB/compare_verb_results_llm.json"

    # 加载模型
    tokenizer, model = load_model(model_path)
    print("模型加载成功！")

    # 加载测试单词
    test_words = load_test_words(json_path)
    
    print("\n=== 开始处理单词 ===")
    results = process_all_words(model, tokenizer, test_words, output_path, save_interval=10)
    
    # 最终保存结果
    invalid_output_path = output_path.replace('.json', '_invalid.json')
    stats = save_results(results, output_path, invalid_output_path)
    
    # 打印处理结果
    print("\n=== 处理结果 ===")
    for word, misspellings in results.items():
        if misspellings:
            print(f"{word}: {misspellings}")
        else:
            print(f"{word}: 未能生成变体")
    
    # 打印统计信息
    total_words = len(test_words)
    
    print("\n=== 最终统计 ===")
    print(f"总单词数: {total_words}")
    print(f"成功生成数: {stats['success']}")
    print(f"未能生成数: {stats['failed']}")
    print(f"成功率: {(stats['success']/total_words)*100:.2f}%")
