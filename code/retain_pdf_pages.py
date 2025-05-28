import fitz  # PyMuPDF

def retain_pages(input_path: str, 
                output_path: str,
                start_page: int, 
                end_page: int,
                user_numbering: bool = True) -> None:
    """
    保留指定页码范围（支持用户直观页码或编程页码）
    
    参数：
    input_path: 输入PDF路径
    output_path: 输出PDF路径
    start_page: 起始页码（用户看到的页码，默认从1开始）
    end_page: 结束页码（包含该页）
    user_numbering: 是否使用用户看到的页码系统（默认True，即从1开始）
    """
    doc = fitz.open(input_path)
    total_pages = doc.page_count
    
    # 页码转换逻辑
    if user_numbering:
        # 转换为零基索引
        start = max(start_page - 1, 0)
        end = min(end_page - 1, total_pages - 1)
    else:
        start = max(start_page, 0)
        end = min(end_page, total_pages - 1)
    
    # 创建页面选择映射
    page_map = list(range(start, end + 1))
    
    # 执行页面选择
    doc.select(page_map)
    
    # 修正后的保存参数（移除非必要加密参数）
    save_options = {
        "garbage": 4,          # 彻底清理未引用对象
        "deflate": True,        # 压缩内容流
        "linear": True,         # 优化线性阅读
        "pretty": True,         # 优化文件结构
        # "encryption": None,   # 错误参数已移除
        "clean": True,          # 修复交叉引用表
        "preserve_metadata": True  # 保留所有元数据
    }
    
    doc.save(output_path, **save_options)
    doc.close()


def pdf_to_txt(pdf_file, output_txt_file):
    # 打开 PDF 文件
    with fitz.open(pdf_file) as pdf_document:
        with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
            # 遍历每一页
            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                text = page.get_text()  # 提取文本
                txt_file.write(text)
                txt_file.write('\n')  # 每页换行

output_pdf="D:/postgraduated/experiments_related/text_analysis/PrimaFacie_text.pdf"
# 保留看到的第10-96页（示例调用）
retain_pages("D:/library/Prima Facie.pdf", 
           output_pdf,
            start_page=10, 
            end_page=96,
            user_numbering=True)

# 调用函数
output_txt="D:/postgraduated/experiments_related/text_analysis/PrimaFacie_analysis/raw_data/PrimaFacie_text.txt"
pdf_to_txt(output_pdf, output_txt)