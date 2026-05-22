import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


class RFPTextCleaner:
    def __init__(self, config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["preprocessing"]["chunk_size"],
            chunk_overlap=config["preprocessing"]["chunk_overlap"],
            separators=[
                "\n\n",
                r"\n[0-9]+\. ",
                r"\n[가-힣]\. ",
                "\n-",
                "\n",
                " ",
            ],
            is_separator_regex=True,
        )

    def clean_text(self, text):
        """텍스트 노이즈와 과도한 공백을 정리한다."""
        if not text:
            return ""

        text = re.sub(r"\\U[0-9a-fA-F]{8}", "", text)
        text = text.replace("<?>", "").replace("<그림>", "")
        return " ".join(text.split()).strip()

    def _split_with_overlap(self, text, size, overlap):
        if size <= 0:
            return [text]

        step = max(size - overlap, 1)
        return [text[i:i + size] for i in range(0, len(text), step)]

    def run_fixed_size_chunking(self, raw_text, project_name="Unknown project"):
        """
        전체 텍스트를 chunk_size 기준으로 고정 길이 분할한다.
        chunk_overlap이 설정되어 있으면 인접 chunk가 해당 길이만큼 겹친다.
        """
        cleaned = self.clean_text(raw_text)
        min_len = self.config["preprocessing"].get("min_chunk_len", 10)
        if len(cleaned) < min_len:
            return []

        size = self.config["preprocessing"]["chunk_size"]
        overlap = self.config["preprocessing"].get("chunk_overlap", 0)
        raw_chunks = self._split_with_overlap(cleaned, size, overlap)

        return [f"[{project_name}] {chunk}" for chunk in raw_chunks if len(chunk) >= min_len]

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
        Markdown heading 기준으로 섹션을 나누고, 긴 섹션은 chunk_size 기준으로 추가 분할한다.
        """
        min_len = self.config["preprocessing"].get("min_chunk_len", 10)
        size = self.config["preprocessing"]["chunk_size"]
        overlap = self.config["preprocessing"].get("chunk_overlap", 0)

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

    def run_semantic_chunking(self, base_chunks, project_name="Unknown project"):
        """
        RecursiveCharacterTextSplitter를 사용해 separator 우선순위를 고려하며 분할한다.
        base_chunks에는 문자열 리스트나 text 속성을 가진 객체 리스트를 넣을 수 있다.
        """
        min_len = self.config["preprocessing"].get("min_chunk_len", 10)
        final_results = []

        for chunk in base_chunks:
            raw_text = chunk.text if hasattr(chunk, "text") else chunk
            cleaned = self.clean_text(raw_text)
            if len(cleaned) < min_len:
                continue

            chunk_project_name = getattr(chunk, "project_name", project_name)
            for split_text in self.splitter.split_text(cleaned):
                if len(split_text) >= min_len:
                    final_results.append(f"[{chunk_project_name}] {split_text}")

        return final_results
