import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RFPTextCleaner:
    def __init__(self, config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['preprocessing']['chunk_size'],
            chunk_overlap=config['preprocessing']['chunk_overlap'],
            separators=["\n\n", r"\n[0-9]+\. ", r"\n[가-힣]\. ", r"\n○", r"\n□", r"\n-", "\n", " "],
            is_separator_regex=True
        )

    def clean_text(self, text):
        """유니코드 노이즈 및 특수 태그 제거"""
        text = re.sub(r'\\U[0-9a-fA-F]{8}', '', text) 
        text = text.replace('<표>', '').replace('<그림>', '')
        return " ".join(text.split()).strip()

    def run_semantic_chunking(self, base_chunks):
        """기존 청크를 의미 단위로 재분할 및 사업명 주입"""
        final_results = []
        for chunk in base_chunks:
            raw_text = chunk.text if hasattr(chunk, 'text') else chunk
            cleaned = self.clean_text(raw_text)
            
            if len(cleaned) < self.config['preprocessing']['min_chunk_len']:
                continue
                
            splits = self.splitter.split_text(cleaned)
            project_name = getattr(chunk, '사업명', '알 수 없는 사업')
            
            for s in splits:
                final_results.append(f"[{project_name}] {s}")
        return final_results