"""报告生成工具。"""

import json
import time
from pathlib import Path

import pandas as pd


def save_predictions_to_csv(results, output_path):
    """将推理结果保存为 CSV 文件。
    
    参数:
        results: 推理结果列表，每个元素为包含 path/prob/label/label_name 的字典
        output_path: 输出文件路径（str 或 Path）
        
    返回:
        str: 写入完成后的绝对路径
    """
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return str(output_path.resolve())


def save_predictions_to_json(results, summary, output_path, class_names=None):
    """将推理结果保存为 JSON 文件。
    
    参数:
        results: 推理结果列表
        summary: 推理摘要信息字典
        output_path: 输出文件路径（str 或 Path）
        class_names: 类别名称列表（可选）
        
    返回:
        str: 写入完成后的绝对路径
    """
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": summary,
        "class_names": class_names or [],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(output_path.resolve())