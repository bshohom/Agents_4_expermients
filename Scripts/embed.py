from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


WHITESPACE_RE = re.compile(r"\s+")
import os
import sys
import signal
import time

STOP_REQUESTED = False
deadline_epoch: float | None = None

def _handle_term(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

def should_stop(deadline_epoch: float | None) -> bool:
    if STOP_REQUESTED:
        return True
    if deadline_epoch is None:
        return False
    return time.time() >= deadline_epoch

print("hello from cluster")
print("python executable:", sys.executable)
print("ELSEVIER_APIKEY exists:", "ELSEVIER_APIKEY" in os.environ)

@dataclass
class ChunkRecord:
    source_path: str
    file_name: str
    title: str
    chunk_id: int
    chunk_start: int
    chunk_end: int
    char_length: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed every PDF under lit/ as chunked documents."
    )
    parser.add_argument(
        "--lit-dir",
        type=Path,
        default=Path("lit"),
        help="Directory that contains the literature PDFs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("lit_embedding_store"),
        help="Output directory for chunk metadata and persisted vector store.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1800,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name for HuggingFaceEmbeddings.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip tiny chunks shorter than this many characters.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Maximum seconds to allow the script to run before exiting. Useful for testing with limited cluster job time. By default, there is no time limit.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pypdf\n"
            "Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)

    return normalize_text("\n".join(pages))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[tuple[int, int, str]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield start, end, chunk
        if end >= len(text):
            break


def build_embedding_model(model_name: str) -> LangchainEmbedding:
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"

    return LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def title_from_path(pdf_path: Path) -> str:
    return pdf_path.stem.replace("_", " ")


def collect_documents(
    lit_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
) -> tuple[list[Document], list[ChunkRecord], list[dict[str, str]], int]:
    pdf_paths = sorted(path for path in lit_dir.rglob("*.pdf") if path.is_file())
    documents: list[Document] = []
    chunk_records: list[ChunkRecord] = []
    failures: list[dict[str, str]] = []

    for pdf_idx, pdf_path in enumerate(pdf_paths, start=1):
        if should_stop(deadline_epoch):
            break
        print(f"[{pdf_idx}/{len(pdf_paths)}] Reading {pdf_path}")

        try:
            text = extract_pdf_text(pdf_path)
        except Exception as exc:  # pragma: no cover
            failures.append({"path": str(pdf_path), "error": str(exc)})
            print(f"  failed: {exc}")
            continue

        if not text:
            failures.append({"path": str(pdf_path), "error": "empty_text"})
            print("  skipped: empty text")
            continue

        title = title_from_path(pdf_path)
        chunk_count = 0

        for chunk_id, (start, end, chunk) in enumerate(
            chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            if len(chunk) < min_chars:
                continue

            metadata = {
                "source_path": str(pdf_path),
                "file_name": pdf_path.name,
                "title": title,
                "chunk_id": chunk_id,
                "chunk_start": start,
                "chunk_end": end,
                "char_length": len(chunk),
            }
            documents.append(Document(text=chunk, metadata=metadata))
            chunk_records.append(ChunkRecord(text=chunk, **metadata))
            chunk_count += 1

        print(f"  chunks kept: {chunk_count}")

    return documents, chunk_records, failures, len(pdf_paths)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_processed_source_paths(chunks_path: Path, failures_path: Path) -> set[str]:
    processed: set[str] = set()

    for row in read_jsonl(chunks_path):
        source_path = str(row.get("source_path", "")).strip()
        if source_path:
            processed.add(source_path)

    for row in read_jsonl(failures_path):
        source_path = str(row.get("path", "")).strip()
        if source_path:
            processed.add(source_path)

    return processed


def load_chunk_documents(chunks_path: Path) -> tuple[list[Document], list[ChunkRecord]]:
    documents: list[Document] = []
    chunk_records: list[ChunkRecord] = []

    for row in read_jsonl(chunks_path):
        text = row.get("text", "")
        metadata = {
            "source_path": row.get("source_path", ""),
            "file_name": row.get("file_name", ""),
            "title": row.get("title", ""),
            "chunk_id": row.get("chunk_id", -1),
            "chunk_start": row.get("chunk_start", -1),
            "chunk_end": row.get("chunk_end", -1),
            "char_length": row.get("char_length", len(text)),
        }
        documents.append(Document(text=text, metadata=metadata))
        chunk_records.append(ChunkRecord(text=text, **metadata))

    return documents, chunk_records


def write_summary(
    summary_path: Path,
    *,
    lit_dir: Path,
    pdf_count: int,
    processed_pdf_count: int,
    chunk_count: int,
    failure_count: int,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    storage_dir: Path,
    chunks_path: Path,
    failures_path: Path,
    completed_all_pdfs: bool,
    index_built: bool,
    timed_out: bool,
) -> dict:
    summary = {
        "lit_dir": str(lit_dir),
        "pdf_count": pdf_count,
        "processed_pdf_count": processed_pdf_count,
        "chunk_count": chunk_count,
        "failure_count": failure_count,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chars": min_chars,
        "storage_dir": str(storage_dir),
        "chunks_path": str(chunks_path),
        "failures_path": str(failures_path),
        "completed_all_pdfs": completed_all_pdfs,
        "index_built": index_built,
        "timed_out": timed_out,
        "stop_requested": STOP_REQUESTED,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    lit_dir = args.lit_dir.resolve()
    out_dir = args.out_dir.resolve()
    global deadline_epoch
    signal.signal(signal.SIGTERM, _handle_term)

    deadline_epoch = None
    if args.max_seconds is not None:
        deadline_epoch = time.time() + args.max_seconds
    if not lit_dir.exists():
        raise SystemExit(f"lit directory does not exist: {lit_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = out_dir / "chunks.jsonl"
    failures_path = out_dir / "failures.jsonl"
    storage_dir = out_dir / "storage"
    summary_path = out_dir / "summary.json"

    pdf_paths = sorted(path for path in lit_dir.rglob("*.pdf") if path.is_file())
    processed_source_paths = load_processed_source_paths(chunks_path, failures_path)
    completed_now = 0

    for pdf_idx, pdf_path in enumerate(pdf_paths, start=1):
        pdf_key = str(pdf_path.resolve())
        if pdf_key in processed_source_paths:
            print(f"[{pdf_idx}/{len(pdf_paths)}] Skipping already processed {pdf_path}")
            continue

        if should_stop(deadline_epoch):
            print("Stopping before next PDF due to timeout or termination request.")
            break

        print(f"[{pdf_idx}/{len(pdf_paths)}] Reading {pdf_path}")

        try:
            text = extract_pdf_text(pdf_path)
        except Exception as exc:  # pragma: no cover
            failure_row = {"path": pdf_key, "error": str(exc)}
            append_jsonl(failures_path, failure_row)
            processed_source_paths.add(pdf_key)
            completed_now += 1
            print(f"  failed: {exc}")
            write_summary(
                summary_path,
                lit_dir=lit_dir,
                pdf_count=len(pdf_paths),
                processed_pdf_count=len(processed_source_paths),
                chunk_count=len(read_jsonl(chunks_path)),
                failure_count=len(read_jsonl(failures_path)),
                embedding_model=args.model_name,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
                storage_dir=storage_dir,
                chunks_path=chunks_path,
                failures_path=failures_path,
                completed_all_pdfs=len(processed_source_paths) >= len(pdf_paths),
                index_built=storage_dir.exists(),
                timed_out=should_stop(deadline_epoch),
            )
            continue

        if not text:
            append_jsonl(failures_path, {"path": pdf_key, "error": "empty_text"})
            processed_source_paths.add(pdf_key)
            completed_now += 1
            print("  skipped: empty text")
            write_summary(
                summary_path,
                lit_dir=lit_dir,
                pdf_count=len(pdf_paths),
                processed_pdf_count=len(processed_source_paths),
                chunk_count=len(read_jsonl(chunks_path)),
                failure_count=len(read_jsonl(failures_path)),
                embedding_model=args.model_name,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
                storage_dir=storage_dir,
                chunks_path=chunks_path,
                failures_path=failures_path,
                completed_all_pdfs=len(processed_source_paths) >= len(pdf_paths),
                index_built=storage_dir.exists(),
                timed_out=should_stop(deadline_epoch),
            )
            continue

        title = title_from_path(pdf_path)
        chunk_count_for_pdf = 0

        for chunk_id, (start, end, chunk) in enumerate(
            chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        ):
            if len(chunk) < args.min_chars:
                continue

            record = ChunkRecord(
                source_path=pdf_key,
                file_name=pdf_path.name,
                title=title,
                chunk_id=chunk_id,
                chunk_start=start,
                chunk_end=end,
                char_length=len(chunk),
                text=chunk,
            )
            append_jsonl(chunks_path, asdict(record))
            chunk_count_for_pdf += 1

        processed_source_paths.add(pdf_key)
        completed_now += 1
        print(f"  chunks kept: {chunk_count_for_pdf}")

        write_summary(
            summary_path,
            lit_dir=lit_dir,
            pdf_count=len(pdf_paths),
            processed_pdf_count=len(processed_source_paths),
            chunk_count=len(read_jsonl(chunks_path)),
            failure_count=len(read_jsonl(failures_path)),
            embedding_model=args.model_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chars=args.min_chars,
            storage_dir=storage_dir,
            chunks_path=chunks_path,
            failures_path=failures_path,
            completed_all_pdfs=len(processed_source_paths) >= len(pdf_paths),
            index_built=storage_dir.exists(),
            timed_out=should_stop(deadline_epoch),
        )

    documents, chunk_records = load_chunk_documents(chunks_path)
    failures = read_jsonl(failures_path)

    if not documents:
        raise SystemExit("No chunked documents were created.")

    completed_all_pdfs = len(processed_source_paths) >= len(pdf_paths)
    index_built = False

    if completed_all_pdfs and not should_stop(deadline_epoch):
        Settings.embed_model = build_embedding_model(args.model_name)
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=str(storage_dir))
        index_built = True
    else:
        print("Skipping vector store rebuild because processing stopped before all PDFs completed.")

    summary = write_summary(
        summary_path,
        lit_dir=lit_dir,
        pdf_count=len(pdf_paths),
        processed_pdf_count=len(processed_source_paths),
        chunk_count=len(chunk_records),
        failure_count=len(failures),
        embedding_model=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
        storage_dir=storage_dir,
        chunks_path=chunks_path,
        failures_path=failures_path,
        completed_all_pdfs=completed_all_pdfs,
        index_built=index_built,
        timed_out=should_stop(deadline_epoch),
    )

    print("\nEmbedding complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
