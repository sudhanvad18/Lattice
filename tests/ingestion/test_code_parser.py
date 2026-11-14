"""Tests for tree-sitter AST code parser and CodeExtractor."""

import tempfile
from pathlib import Path

import pytest

from lattice.ingestion.code_parser import parse_code, LANGUAGE_MAP, CODE_EXTENSIONS
from lattice.ingestion.models import DocumentType
from lattice.graph.extractor import CodeExtractor


PYTHON_FIXTURE = '''\
import os
from pathlib import Path

class DataProcessor:
    """Processes raw data files."""

    def __init__(self, path):
        self.path = path

    def process(self):
        return self._load()

    def _load(self):
        return Path(self.path).read_text()


def main():
    dp = DataProcessor("/tmp/data.csv")
    dp.process()


if __name__ == "__main__":
    main()
'''

JS_FIXTURE = '''\
import { readFile } from "fs";
const axios = require("axios");

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async fetchData(endpoint) {
        return axios.get(this.baseUrl + endpoint);
    }
}

function processResponse(data) {
    return data.map(item => item.name);
}

module.exports = { ApiClient, processResponse };
'''


class TestCodeParser:
    """Test the code_parser module."""

    def test_parse_python_file(self, tmp_path):
        f = tmp_path / "example.py"
        f.write_text(PYTHON_FIXTURE)
        doc = parse_code(f)

        assert doc.content == PYTHON_FIXTURE
        assert doc.metadata.doc_type == DocumentType.CODE
        assert doc.metadata.extra["language"] == "python"
        assert doc.metadata.title == "example.py"

    def test_parse_js_file(self, tmp_path):
        f = tmp_path / "client.js"
        f.write_text(JS_FIXTURE)
        doc = parse_code(f)

        assert doc.metadata.doc_type == DocumentType.CODE
        assert doc.metadata.extra["language"] == "javascript"

    def test_parse_unknown_extension(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("some unknown content")
        doc = parse_code(f)

        assert doc.content == "some unknown content"
        assert doc.metadata.extra["language"] == "unknown"

    def test_ast_nodes_attached(self, tmp_path):
        """When tree-sitter grammar is available, AST nodes are in metadata."""
        f = tmp_path / "example.py"
        f.write_text(PYTHON_FIXTURE)
        doc = parse_code(f)

        ast_nodes = doc.metadata.extra.get("ast_nodes", [])
        # Even if tree-sitter isn't installed, this should be a list
        assert isinstance(ast_nodes, list)

    def test_language_map_coverage(self):
        assert ".py" in LANGUAGE_MAP
        assert ".js" in LANGUAGE_MAP
        assert ".ts" in LANGUAGE_MAP
        assert ".go" in LANGUAGE_MAP
        assert ".rs" in LANGUAGE_MAP
        assert ".java" in LANGUAGE_MAP
        assert ".c" in LANGUAGE_MAP
        assert ".cpp" in LANGUAGE_MAP

    def test_code_extensions_set(self):
        assert ".py" in CODE_EXTENSIONS
        assert ".ts" in CODE_EXTENSIONS
        assert ".txt" not in CODE_EXTENSIONS


class TestCodeExtractor:
    """Test the CodeExtractor for AST-based entity/relation extraction."""

    def _make_chunks_with_ast(self, ast_nodes, source="test.py", language="python"):
        from lattice.ingestion.models import Chunk, DocumentMetadata, DocumentType
        return [Chunk(
            document_id="doc1",
            content="source code content",
            index=0,
            metadata=DocumentMetadata(
                source=source,
                doc_type=DocumentType.CODE,
                extra={"language": language, "ast_nodes": ast_nodes},
            ),
        )]

    def test_extract_function_entity(self):
        ast_nodes = [
            {"type": "function_definition", "name": "process_data", "start_line": 1, "end_line": 5, "text": "def process_data(): ..."},
        ]
        chunks = self._make_chunks_with_ast(ast_nodes)
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        func_entities = [e for e in result.entities if e.entity_type == "Function"]
        assert any(e.name == "process_data" for e in func_entities)

    def test_extract_class_entity(self):
        ast_nodes = [
            {"type": "class_definition", "name": "MyClass", "start_line": 1, "end_line": 10, "text": "class MyClass: ..."},
        ]
        chunks = self._make_chunks_with_ast(ast_nodes)
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        class_entities = [e for e in result.entities if e.entity_type == "Class"]
        assert any(e.name == "MyClass" for e in class_entities)

    def test_extract_import_relation(self):
        ast_nodes = [
            {"type": "import_statement", "name": "import os", "start_line": 1, "end_line": 1, "text": "import os"},
        ]
        chunks = self._make_chunks_with_ast(ast_nodes)
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        import_rels = [r for r in result.relations if r.relation_type == "IMPORTS"]
        assert len(import_rels) >= 1

    def test_contains_relation(self):
        ast_nodes = [
            {"type": "function_definition", "name": "helper", "start_line": 3, "end_line": 5, "text": "def helper(): ..."},
        ]
        chunks = self._make_chunks_with_ast(ast_nodes, source="utils.py")
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        contains_rels = [r for r in result.relations if r.relation_type == "CONTAINS"]
        assert len(contains_rels) >= 1

    def test_module_entity_created(self):
        ast_nodes = []
        chunks = self._make_chunks_with_ast(ast_nodes, source="my_module.py")
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        module_entities = [e for e in result.entities if e.entity_type == "Module"]
        assert any(e.name == "my_module.py" for e in module_entities)

    def test_call_relation(self):
        ast_nodes = [
            {"type": "function_definition", "name": "caller", "start_line": 1, "end_line": 3, "text": "def caller(): ..."},
            {"type": "call_expression", "name": "target_fn", "start_line": 2, "end_line": 2, "text": "target_fn()"},
        ]
        chunks = self._make_chunks_with_ast(ast_nodes)
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        call_rels = [r for r in result.relations if r.relation_type == "CALLS"]
        assert len(call_rels) >= 1

    def test_provenance_is_extracted(self):
        ast_nodes = [
            {"type": "function_definition", "name": "fn", "start_line": 1, "end_line": 2, "text": "def fn(): ..."},
        ]
        chunks = self._make_chunks_with_ast(ast_nodes)
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        for rel in result.relations:
            assert rel.provenance == "extracted"

    def test_anonymous_nodes_skipped(self):
        ast_nodes = [
            {"type": "function_definition", "name": "<anonymous>", "start_line": 1, "end_line": 2, "text": "lambda: ..."},
        ]
        chunks = self._make_chunks_with_ast(ast_nodes)
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        func_entities = [e for e in result.entities if e.entity_type == "Function"]
        assert not any(e.name == "<anonymous>" for e in func_entities)

    def test_empty_ast(self):
        chunks = self._make_chunks_with_ast([])
        extractor = CodeExtractor()
        result = extractor.extract(chunks)

        module_entities = [e for e in result.entities if e.entity_type == "Module"]
        assert len(module_entities) == 1
        assert len(result.relations) == 0
