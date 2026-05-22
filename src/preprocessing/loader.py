import pymupdf4llm
import time
import os

def extract_pdf(filepath, pages=None, image_path="./images", write_images=False):
    """
    PDF 파일을 읽어서 마크다운 텍스트로 추출하는 함수
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {filepath}")
        
    start_time = time.time()
    
    md_text = pymupdf4llm.to_markdown(
        filepath,
        pages=pages,
        write_images=write_images,
        image_path=image_path,
    )
    
    elapsed = time.time() - start_time
    print(f"📄 {os.path.basename(filepath)} 추출 완료 ({elapsed:.3f}초 소요)")
    return md_text