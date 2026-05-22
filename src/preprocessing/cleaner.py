import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RFPTextCleaner:
    def __init__(self, config):
        self.config = config
        # 팀원의 랭체인 텍스트 스플리터 세팅
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['preprocessing']['chunk_size'],
            chunk_overlap=config['preprocessing']['chunk_overlap'],
            separators=["\n\n", r"\n[0-9]+\. ", r"\n[가-힣]\. ", r"\n○", r"\n□", r"\n-", "\n", " "],
            is_separator_regex=True
        )

    def clean_text(self, text):
        """유니코드 노이즈 및 특수 태그 제거"""
        if not text:
            return ""
        text = re.sub(r'\\U[0-9a-fA-F]{8}', '', text) 
        text = text.replace('<표>', '').replace('<그림>', '')
        # 공백 정제 (주의: 고정 크기 청킹 시 줄바꿈을 다 지우고 1자로 만들지, 줄바꿈을 살릴지에 따라 선택 가능)
        return " ".join(text.split()).strip()

    def run_fixed_size_chunking(self, raw_text, project_name="알 수 없는 사업"):
        """
        오버랩 없이 정확히 지정된 글자 수(chunk_size)만큼 슬라이싱하고 사업명 주입
        """
        # 1. 먼저 기존에 정의된 clean_text로 노이즈 정제
        cleaned = self.clean_text(raw_text)
        
        # 2. 설정된 최소 길이보다 작으면 버림
        min_len = self.config['preprocessing'].get('min_chunk_len', 10)
        if len(cleaned) < min_len:
            return []
            
        # 3. 설정파일(config.yaml)에서 chunk_size 가져오기 (기본값 500)
        size = self.config['preprocessing']['chunk_size']
        
        # 4. 순수 파이썬 슬라이싱으로 정확하게 칼같이 자르기
        raw_chunks = [cleaned[i:i + size] for i in range(0, len(cleaned), size)]
        
        # 5. 비교 실험 메타데이터 관리를 위해 앞에 [사업명] 주입하기
        final_results = [f"[{project_name}] {chunk}" for chunk in raw_chunks]
        return final_results

    def _split_with_overlap(self, text, size, overlap):
        if size <= 0:
            return [text]

        step = max(size - overlap, 1)
        return [text[i:i + size] for i in range(0, len(text), step)]

    def _split_markdown_sections(self, markdown_text):
        sections = []
        current_title = ""
        current_lines = []
        header_stack = []

        for line in markdown_text.splitlines():
            header_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if header_match:
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                    current_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                header_stack = header_stack[:level - 1]
                header_stack.append(title)
                current_title = " > ".join(header_stack)
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        return sections

    def run_markdown_chunking(self, markdown_text, project_name="Unknown project"):
        """
        Split Markdown by heading sections, then split long sections by chunk_size.
        """
        min_len = self.config['preprocessing'].get('min_chunk_len', 10)
        size = self.config['preprocessing']['chunk_size']
        overlap = self.config['preprocessing'].get('chunk_overlap', 0)

        final_results = []
        for section_title, section_text in self._split_markdown_sections(markdown_text):
            cleaned = self.clean_text(section_text)
            if len(cleaned) < min_len:
                continue

            section_prefix = f"[{project_name}]"
            if section_title:
                section_prefix = f"{section_prefix} [{section_title}]"

            for chunk in self._split_with_overlap(cleaned, size, overlap):
                if len(chunk) >= min_len:
                    final_results.append(f"{section_prefix} {chunk}")

        return final_results

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
