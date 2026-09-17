from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


def sliding_window_chunks(text, chunk_size=250, stride=220):
    """把长文本按固定大小切成重叠的小块"""
    return [text[i:i + chunk_size] for i in range(0, len(text), stride)]


def load_pdf(pdf_filename, page_numbers=None):
    """从 PDF 提取文本，切块后返回列表"""
    full_text = ""
    for i, page_layout in enumerate(extract_pages(pdf_filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for ele in page_layout:
            if isinstance(ele, LTTextContainer):
                full_text += ele.get_text().replace('\n', '').replace('\r', '')

    chunks = sliding_window_chunks(full_text, chunk_size=250, stride=220)
    return chunks