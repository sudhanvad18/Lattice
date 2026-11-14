"""
Code file parser using tree-sitter for AST-aware ingestion.

Parses source files into Documents with structural metadata preserved,
so downstream extractors can produce function/class/import entities
and call/import/containment relations from the AST.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import structlog

from lattice.ingestion.models import Document, DocumentMetadata, DocumentType

logger = structlog.get_logger()

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "c",
    ".h": "c",
    ".hpp": "c",
}

GRAMMAR_PACKAGES: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
}


def _get_language(lang_name: str):
    """Load a tree-sitter Language object from its grammar package."""
    pkg_name = GRAMMAR_PACKAGES.get(lang_name)
    if not pkg_name:
        return None
    try:
        import tree_sitter
        mod = importlib.import_module(pkg_name)
        capsule = mod.language_typescript() if lang_name == "typescript" else mod.language()
        return tree_sitter.Language(capsule)
    except (ImportError, AttributeError, TypeError) as e:
        logger.debug("treesitter_grammar_unavailable", language=lang_name, error=str(e))
        return None


def _walk_tree(node, src_bytes: bytes) -> list[dict[str, Any]]:
    """Recursively walk an AST tree and collect structural nodes."""
    results: list[dict[str, Any]] = []
    node_type = node.type

    collect_types = {
        "function_definition", "function_declaration", "method_definition",
        "arrow_function", "function_item",
        "class_definition", "class_declaration", "class_specifier",
        "struct_item", "impl_item",
        "import_statement", "import_from_statement", "import_declaration",
        "use_declaration",
        "call_expression", "call",
        "assignment", "variable_declaration", "let_declaration",
    }

    if node_type in collect_types:
        name = _extract_name(node, src_bytes)
        results.append({
            "type": node_type,
            "name": name,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "text": src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")[:500],
        })

    for child in node.children:
        results.extend(_walk_tree(child, src_bytes))

    return results


def _extract_name(node, src_bytes: bytes) -> str:
    """Extract the name identifier from a structural AST node."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    if node.type in ("import_statement", "import_from_statement", "import_declaration", "use_declaration"):
        return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()[:120]
    return "<anonymous>"


def parse_code(path: Path) -> Document:
    """Parse a source code file using tree-sitter AST analysis.

    Returns a Document whose content is the raw source text, with
    structural AST metadata attached for downstream entity extraction.
    Falls back to plain text if grammar is unavailable.
    """
    path = Path(path)
    ext = path.suffix.lower()
    lang_name = LANGUAGE_MAP.get(ext)

    content = path.read_text(encoding="utf-8", errors="replace")

    ast_nodes: list[dict[str, Any]] = []

    if lang_name:
        language_obj = _get_language(lang_name)
        if language_obj:
            try:
                import tree_sitter

                parser = tree_sitter.Parser()
                parser.language = language_obj
                src_bytes = content.encode("utf-8")
                tree = parser.parse(src_bytes)
                ast_nodes = _walk_tree(tree.root_node, src_bytes)

                logger.info(
                    "code_parsed",
                    path=str(path),
                    language=lang_name,
                    ast_nodes=len(ast_nodes),
                )
            except Exception as e:
                logger.warning("treesitter_parse_failed", path=str(path), error=str(e))

    metadata = DocumentMetadata(
        source=str(path),
        title=path.name,
        doc_type=DocumentType.CODE,
        extra={
            "language": lang_name or "unknown",
            "ast_nodes": ast_nodes,
        },
    )

    return Document(content=content, metadata=metadata)


CODE_EXTENSIONS = set(LANGUAGE_MAP.keys())
