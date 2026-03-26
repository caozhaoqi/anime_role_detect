import PyPDF2

def extract_pdf_text(pdf_path):
    """提取PDF文档的文本内容"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"PDF文档页数: {num_pages}")
            
            text = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n=== 第 {page_num + 1} 页 ===\n"
                    text += page_text
            
            return text
    except Exception as e:
        print(f"提取PDF文本时出错: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/docs/0508.环境部署专题-v45-20260104_144012.pdf"
    text = extract_pdf_text(pdf_path)
    if text:
        # 保存提取的文本到文件
        output_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/docs/0508.环境部署专题-v45-20260104_144012.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"文本已提取并保存到: {output_path}")
        # 打印前1000个字符作为预览
        print("\n=== 文本预览 ===")
        print(text[:1000])
