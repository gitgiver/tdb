import os
import fitz  # PyMuPDF
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import pytesseract
from PIL import Image
import io
import re
import jieba
import jieba.posseg as pseg
from PIL import Image
import pytesseract

# 配置 Tesseract 路径（根据实际安装位置修改）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def extract_text_with_pymupdf(pdf_path, page_num):
    """用 PyMuPDF 提取单页文本"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    return page.get_text()


def extract_text_with_pytesseract(img):
    """用 pytesseract 对图像进行 OCR"""
    img = img.convert('L')  # 转为灰度图
    text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6')
    return text


def split_page_to_two_halves(pdf_path, page_num):
    """将PDF页面切割为上下两部分，返回下半部分的PIL图像"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)

    # 获取页面尺寸
    rect = page.rect
    width, height = rect.width, rect.height

    # 定义下半部分的矩形区域（从垂直中点开始）
    lower_half_rect = fitz.Rect(0, height / 2, width, height)

    # 渲染下半部分为图像
    pix = page.get_pixmap(clip=lower_half_rect, dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes()))
    return img


def clean_newlines(text):
    """
        修正后的清洗逻辑（按处理顺序排列）：
        1. 首先处理所有空白字符组合（空格/tab/换行混合）
        2. 修复被错误换行打断的句子
        3. 保护结构性换行（标题/列表/图表）
        4. 最终合并剩余多余换行
        """
    # 阶段1：标准化所有空白字符 → 单个换行
    text = re.sub(r'[ \t\r\f]*(?:\n[ \t\r\f]*)+', '\n', text)

    # 阶段2：修复中文标点后的错误换行
    text = re.sub(r'([，。；：？！])\n', r'\1', text)

    # 阶段3：保护结构性换行（标题/列表/图表）
    # 3.1 保护章节标题（如"一、XXX\n"）
    text = re.sub(r'([一二三四五六七八九十]、[^\n]+)(?:\n)+', r'\1\n', text)
    # 3.2 保护列表项（如"1、XXX\n"）
    text = re.sub(r'(\d+[\.、][^\n]+)(?:\n)+', r'\1\n', text)
    # 3.3 保护图表标记（如"△XXX\n"）
    text = re.sub(r'(△[^\n]+)(?:\n)+', r'\1\n', text)

    # 阶段4：最终合并多余换行（保留最多1个空行）
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def extract_text_from_pdf(pdf_path):
    """混合提取：
    - 第一页下半部分：OCR
    - 其余部分（包括第一页上半部分）：PyMuPDF
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    # 第一页：上半部分用 PyMuPDF
    first_page_text = extract_text_with_pymupdf(pdf_path, 0)

    # 用正则分割第一页文本（假设上半部分和下半部分有明显分隔）
    # 如果没有分隔符，则按行数粗略分割（示例按前50%行）
    lines = first_page_text.split('\n')
    upper_half_text = '\n'.join(lines[:len(lines) // 2])
    full_text += upper_half_text + "\n"

    # 第一页：下半部分用 OCR
    lower_half_img = split_page_to_two_halves(pdf_path, 0)
    lower_half_text = extract_text_with_pytesseract(lower_half_img)
    full_text += "=== OCR 下半部分 ===\n" + lower_half_text + "\n"

    # 剩余页用 PyMuPDF
    for page_num in range(0, len(doc)):
        page_text = extract_text_with_pymupdf(pdf_path, page_num)
        full_text += page_text + "\n"

    # 处理文本内容
    fullText = full_text
    full_text = clean_newlines(full_text)

    with open(fr'txts\{pdf_path[4:-4]}.txt', 'w', encoding='utf-8') as file:
        file.write(full_text)
    return fullText

def extract_organization(text):
    """提取组织单位（结合正则 + jieba机构名词识别）"""
    # 方法1：正则匹配标准机构名
    org_patterns = [
        r'中国[\w·\-]+(?:服务中心|中心|协会|委员会|办公室|研究院)',
        r'(泰迪杯[^，。\n]+?委员会)'
    ]

    # 尝试所有正则模式
    for org_pattern in org_patterns:
        org_match = re.search(org_pattern, text)
        if org_match:
            # 如果是第二个模式匹配到结果，需要检查是否是"泰迪杯"相关
            if org_pattern == org_patterns[1] and "泰迪杯" not in org_match.group():
                continue  # 如果不是泰迪杯相关，继续尝试其他模式
            return org_match.group()

    # 方法2：用jieba提取机构名词
    words = pseg.cut(text)
    orgs = [word for word, flag in words if flag == 'nt' and '中国' in word]
    return orgs[0] if orgs else None


def extract_publish_date(text):
    """提取发布时间（结合正则 + 语义位置）"""
    date_patterns = [
        r"([^\n]+?(?:委员会|组委会|办公室))\s*[\r\n，]+\s*(\d{4}年\d{1,2}月\d{1,2}日)",  # 模式1：带机构上下文的日期
        r"(?:发布时间|日期|发布日期)[：:\s]*(\d{4}\s*[年\-]\s*\d{1,2}\s*[月])",  # 模式2：直接匹配日期
        r"^.*?(\d{4}\s*年\s*\d{1,2}\s*月)"  # 模式3：行首日期
    ]

    for date_pattern in date_patterns:
        date_match = re.search(date_pattern, text, re.M)
        if not date_match:
            continue

        # 通用处理逻辑：取最后一个非空捕获组
        groups = [g for g in date_match.groups() if g is not None]
        if not groups:
            continue

        date = groups[-1].replace(" ", "")
        if date:
            return date  # 找到有效日期立即返回

    # 正则匹配失败时使用jieba分词
    words = pseg.cut(text)
    dates = [word for word, flag in words if flag == 't' and '年' in word and '月' in word]
    return dates[0] if dates else None

def extract_competition_info(text: str) -> Dict[str, Optional[str]]:
    """从文本中提取竞赛信息"""
    result = {
        'competition_name': None,
        'track': None,
        'publish_time': None,
        'registration_time': None,
        'organizing_unit': None,
        'official_website': None
    }

    # 提取竞赛名称
    name_patterns = [
        r'(\d{4}\s*年（第\d+\s*届）[“”](.*?)挑战赛)',
        r"第(.)[\s\S]*?挑战赛"
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, text)
        if name_match:
            if len(name_match.group(1).strip()) > 5:
                result['competition_name'] = re.sub(r'\s+', '', name_match.group(1).strip())
            else:
                result['competition_name'] = name_match.group(0).strip().split(" ")[0].split('“')[0]
            break

        # 提取赛道信息 - 改进的提取逻辑
    track_patterns = [
            r"专项赛名称[:：]\s*([^\n]+)",
            r"第七届全国青少年人工智能创新挑战赛\s*([^\n]+)\s*专项赛",
            r"([^\n]+)专项赛参赛手册",
            r'(\d{4}\s*年（第\d+\s*届）[“”](.*?)挑战赛)'
        ]
    for pattern in track_patterns:
            track_match = re.search(pattern, text)
            if track_match:
                result['track'] = track_match.group(1).split("）")[-1].strip()
                break

    # 提取发布时间 - 改进的提取逻辑

    result['publish_time'] = extract_publish_date(text)

    # 提取报名时间 - 改进的提取逻辑
    reg_patterns = [
        r"报名.*时间：\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*[-—]\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
        r"报名时间：\s*(\d{4}年\d{1,2}月\d{1,2}日\s*[-—–]\s*\d{1,2}月\d{1,2}日)",
        r"报名时间：\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*[-—–]\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
        r'报名时间\s*[\r\n]+\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*[-~－—]\s*\d{1,2}\s*月\s*\d{1,2}\s*日)',
        r"起始时间：\s*[\r\n]+\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*[-~－—]\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
        r"在(\d{4}\s*年\s*\d{1,2}\s*月前)",
        r"报名截止时间\s*(\d{4}年\d{1,2}月\d{1,2}日\s*[-—–]\s*\d{1,2}月\d{1,2}日)",
        r"报名\s*(\d{4}年\d{1,2}月\d{1,2}日\s*[-—–]\s*\d{1,2}月\d{1,2}日)",
    ]
    for pattern in reg_patterns:
        reg_match = re.search(pattern, text)
        if reg_match:
            result['registration_time'] = re.sub(r'\s+', '', reg_match.group(1))
            break

    # 提取组织单位 - 改进的提取逻辑
    result['organizing_unit'] = extract_organization(text)

    # 提取官网 - 改进的提取逻辑
    website_patterns = [
        r"官网[:：]\s*([^\n]+)",
        r"官方网站[:：]\s*([^\n]+)",
        r"网站[:：]\s*([^\n]+)",
        r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)"
    ]
    for pattern in website_patterns:
        website_match = re.search(pattern, text)
        if website_match:
            url=website_match.group(1)
            url = re.sub(r'[；，""。“”""\u4e00-\u9fa5]', '', url)
            # 去除连续空白符
            url = re.sub(r'\s+', '', url)
            # 去除末尾的标点
            url = re.sub(r'[./""""]+$', '', url)
            # 去除末尾斜杠
            url = re.sub(r'/+$', '', url)
            # 去除端口号（如:8080）
            url = re.sub(r':\d+$', '', url)
            result['official_website'] = re.sub(r'[；，”。"“\u4e00-\u9fa5]', '', re.sub(r'\s+', '', url))
            break

    return result


def save_to_excel(data_list: List[Dict], output_file: str):
    """将数据保存到Excel文件"""
    df = pd.DataFrame(data_list)
    custom_headers = {
        'competition_name': '赛项名称',
        'track': '赛道',
        'publish_time': '发布时间',  # 示例：将字段名改为中文
        'registration_time': '报名时间',
        'organizing_unit':"组织单位",
        "official_website":"官网"
    }
    df.rename(columns=custom_headers, inplace=True)
    df.to_excel(output_file, index=False, engine='openpyxl')


def process_pdf_files(directory: str):
    """处理PDF文件并提取信息"""
    all_data = []

    # 查找匹配的文件
    for f in os.listdir(directory):
        if f.lower().endswith('.pdf'):
            file_path = os.path.join(directory, f)
            try:
                print(f"正在处理文件: {file_path}")
                text = extract_text_from_pdf(file_path)

                # 提取竞赛信息
                competition_data = extract_competition_info(text)


                # 打印提取的数据用于调试
                print(f"提取的竞赛数据: {competition_data}")

                # 添加到列表用于Excel导出
                combined_data = {**competition_data}
                all_data.append(combined_data)

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

    # 保存到Excel
    if all_data:
        save_to_excel(all_data, r"txts\result_1.xlsx")
        print("数据已保存到 txts\result_1.xlsx")


if __name__ == "__main__":
    pdf_directory = "pdfs"  # 替换为你的PDF文件目录
    process_pdf_files(pdf_directory)
