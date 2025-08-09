import asyncio
import requests
import tempfile
import os
import time
import re
from typing import List

import fitz  # PyMuPDF
from pymupdf4llm import to_markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        # We only need the RecursiveCharacterTextSplitter, configured to add the start index
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True # This is crucial for the mapping logic
        )

    def _extract_and_add_bold_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Extracts bold text (text between **) from each chunk's content 
        and adds it to the metadata.
        """
        # Regex to find all text between double asterisks (non-greedy)
        bold_pattern = re.compile(r'\*\*(.*?)\*\*', re.DOTALL)
        
        for chunk in chunks:
            # Find all bold phrases in the chunk's content
            bold_phrases = bold_pattern.findall(chunk.page_content)
            
            if bold_phrases:
                # Clean up phrases (remove newlines, extra spaces) and add to metadata
                cleaned_phrases = [re.sub(r'\s+', ' ', phrase).strip() for phrase in bold_phrases]
                chunk.metadata['important_phrases'] = cleaned_phrases
                
        return chunks

    def load_and_chunk_from_url(self, pdf_url: str) -> List[Document]:
        """
        Downloads, converts to Markdown, chunks by size, and maps hierarchical metadata.
        """
        start_time = time.perf_counter()
        
        # 1. Download and Convert to Markdown
        print(f"Fetching PDF from URL: {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            pdf_filepath = temp_file.name
        
        print("Converting PDF to Markdown...")
        doc = fitz.open(pdf_filepath)
        full_markdown_text = to_markdown(doc)
        doc.close()
        os.remove(pdf_filepath)
        
        # 2. Create a "map" of all headings and their character locations
        headings_map = []
        for match in re.finditer(r'^(#+)\s(.*)', full_markdown_text, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2)
            start_index = match.start()
            headings_map.append({"level": level, "text": text, "start_index": start_index})

        # 3. Perform the size-based chunking
        size_based_chunks = self.text_splitter.create_documents([full_markdown_text])
        
        # 4. Map the headings to each chunk
        final_chunks = []
        for chunk in size_based_chunks:
            chunk_start_index = chunk.metadata.get('start_index', 0)
            current_headers = {}
            
            # Find the most recent headers for this chunk's position
            for heading in headings_map:
                if heading['start_index'] <= chunk_start_index:
                    header_key = f"H{heading['level']}"
                    current_headers[header_key] = heading['text']
                else:
                    break # Stop when we pass the chunk's position
            
            # Clean up lower-level headers
            if "H1" in current_headers:
                if "H2" in current_headers and not current_headers["H2"].startswith(current_headers["H1"]):
                    del current_headers["H2"]
                if "H3" in current_headers and ("H2" not in current_headers or not current_headers["H3"].startswith(current_headers.get("H2", ""))):
                    del current_headers["H3"]
            
            # Add the found hierarchy to the chunk's metadata
            chunk.metadata.update(current_headers)
            del chunk.metadata['start_index'] # Clean up the temporary index
            final_chunks.append(chunk)
            final_chunks_with_bold = self._extract_and_add_bold_metadata(final_chunks)

        print(f"Successfully created {len(final_chunks_with_bold)} structured chunks.")
        return final_chunks_with_bold

    async def load_and_chunk_from_url_async(self, pdf_url: str) -> List[Document]:
        loop = asyncio.get_running_loop()
        documents = await loop.run_in_executor(None, self.load_and_chunk_from_url, pdf_url)
        return documents

# --- Test Execution ---
# The test block can remain the same, but it will now show the new chunk format.
async def main():
    PDF_URL = "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D"
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=150)
    document_chunks = await pdf_processor.load_and_chunk_from_url_async(PDF_URL)

    if document_chunks:
        print("\n--- Verification of Enriched Chunks ---")
        
        start_index = 450
        num_to_display = 3
        
        end_index = min(start_index + num_to_display, len(document_chunks))

        for i in range(start_index, end_index):
            chunk = document_chunks[i]
            print(f"\n--- Chunk {i + 1} ---")
            print(chunk.page_content)
            print(chunk.metadata)

if __name__ == "__main__":
    asyncio.run(main())