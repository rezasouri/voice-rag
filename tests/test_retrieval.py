from pathlib import Path

import pytest

from voice_rag.retrieval import Index, Passage, chunk_text, format_context, passages_from_directory, tokens


def test_normalize_persian_unicode_and_zwnj():
    assert tokens("كتاب‌هاي ايران") == ["کتاب", "های", "ایران"]
    assert "کتاب" in tokens("کِتاب")


def test_index_finds_source_and_is_stable(tmp_path: Path):
    (tmp_path / "policy.txt").write_text(
        "مدت امانت کتاب برای اعضای پایه چهارده روز است و تمدید هفت روز انجام می‌شود.",
        encoding="utf-8",
    )
    (tmp_path / "hours.md").write_text(
        "ساعات پشتیبانی شنبه تا چهارشنبه از نه صبح تا پنج عصر است.",
        encoding="utf-8",
    )
    index = Index.build(tmp_path)
    assert len(index.passages) == 2
    hits = index.search("امانت کتاب چند روز است؟")
    assert hits and hits[0].source == "policy.txt"
    output = tmp_path / "index.json"
    index.save(output)
    assert Index.load(output).search("امانت کتاب چند روز است؟") == hits
    assert "policy.txt" in format_context(hits)


def test_non_matching_query_returns_no_citation():
    index = Index([Passage("a.md", "درباره ساعت کاری و تمدید کتاب")])
    assert index.search("اتوبوس بین شهری") == []
    assert "هیچ سند مرتبطی" in format_context([])


def test_chunk_overlap():
    ch = chunk_text(" ".join(f"w{i}" for i in range(12)), words_per_chunk=6, overlap=2)
    assert len(ch) == 3
    assert ch[0].split()[-2:] == ch[1].split()[:2]
    with pytest.raises(ValueError):
        chunk_text("hello", 2, 2)


def test_semantic_retrieval_with_no_lexical_match():
    idx = Index([Passage("a.md", "ساعت پاسخگویی اداری", (1.0, 0.0)), Passage("b.md", "تمدید کتاب", (0.0, 1.0))])
    assert idx.search("پرسش بدون واژه مشترک", query_embedding=[1.0, 0.0])[0].source == "a.md"


def test_document_ingestion_skips_hidden_and_binary(tmp_path: Path):
    (tmp_path / ".secret.md").write_text("secret message", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"123")
    (tmp_path / "answer.md").write_text("سلام من یک متن بلند در مجموعه آزمایشی هستم", encoding="utf-8")
    docs = passages_from_directory(tmp_path)
    assert len(docs) == 1 and docs[0].source == "answer.md"


def test_empty_and_invalid_index_queries(tmp_path: Path):
    index = Index([])
    assert index.search("سلام") == []
    path = tmp_path / "invalid.json"
    path.write_text("{\"schema_version\": 42, \"passages\": []}", encoding="utf-8")
    with pytest.raises(ValueError):
        Index.load(path)


def test_nested_source_names(tmp_path: Path):
    sub = tmp_path / "farsi"
    sub.mkdir()
    (sub / "notes.md").write_text("عنوان فرایند تمدید کارت کتابخانه است.", encoding="utf-8")
    assert passages_from_directory(tmp_path)[0].source == "farsi/notes.md"
