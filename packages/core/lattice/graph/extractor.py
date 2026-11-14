"""
Entity and relation extraction from document chunks.

Two strategies:
1. Rule-based (default for local demo): uses patterns and heuristics
2. LLM-powered (optional): uses an LLM to extract structured entities

The rule-based extractor handles common patterns well:
- Defined terms ("X is a Y", "X refers to Y")
- Hierarchical relationships (headings → subheadings)
- Technical identifiers (part numbers, spec numbers, document references)
- Named entities (capitalized phrases, acronyms)

Design: extractors produce Entity and Relation objects that the KG builder
adds to the graph. Extraction is decoupled from storage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from lattice.graph.models import Entity, Relation
from lattice.ingestion.models import Chunk

logger = structlog.get_logger()


@dataclass
class ExtractionResult:
    """Entities and relations extracted from a set of chunks."""

    entities: list[Entity]
    relations: list[Relation]

    @property
    def total(self) -> int:
        return len(self.entities) + len(self.relations)


class RuleBasedExtractor:
    """Extracts entities and relations using pattern matching.

    This is the zero-dependency, zero-API-cost option. It won't catch
    everything, but it provides a solid baseline for structured documents.
    """

    # Patterns for common entity indicators
    ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,6})\b")
    DEFINED_TERM_PATTERN = re.compile(
        r"(?:^|\. )([A-Z][a-z]+(?: [A-Z][a-z]+)+)"
    )
    SPEC_NUMBER_PATTERN = re.compile(
        r"\b((?:AMS|ASTM|MIL|SAE|ISO|EMM|SB|AD)\s*[-]?\s*[\d][\w\-\.]*)\b"
    )
    PART_NUMBER_PATTERN = re.compile(r"\b(PN[-\s]?\d{3,}[-\w]*)\b")
    HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    # Relation indicators
    RELATION_PATTERNS = [
        (re.compile(r"(.+?)\s+(?:is a|refers to|means)\s+(.+?)(?:\.|$)", re.IGNORECASE), "IS_A"),
        (re.compile(r"(.+?)\s+(?:requires|needs|demands)\s+(.+?)(?:\.|$)", re.IGNORECASE), "REQUIRES"),
        (re.compile(r"(.+?)\s+(?:causes|leads to|results in)\s+(.+?)(?:\.|$)", re.IGNORECASE), "CAUSES"),
        (re.compile(r"(.+?)\s+(?:per|according to|as defined in)\s+(.+?)(?:\.|$)", re.IGNORECASE), "REFERENCES"),
    ]

    def extract(self, chunks: list[Chunk]) -> ExtractionResult:
        """Extract entities and relations from a list of chunks."""
        entities: dict[str, Entity] = {}  # name → entity (dedup by name)
        relations: list[Relation] = []

        for chunk in chunks:
            chunk_entities = self._extract_entities(chunk)
            for entity in chunk_entities:
                if entity.name in entities:
                    # Merge source chunk IDs
                    existing = entities[entity.name]
                    if chunk.id not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chunk.id)
                else:
                    entities[entity.name] = entity

            chunk_relations = self._extract_relations(chunk, entities)
            relations.extend(chunk_relations)

        logger.info(
            "extraction_complete",
            entities=len(entities),
            relations=len(relations),
            chunks_processed=len(chunks),
        )

        return ExtractionResult(
            entities=list(entities.values()),
            relations=relations,
        )

    def _extract_entities(self, chunk: Chunk) -> list[Entity]:
        """Extract entities from a single chunk."""
        entities: list[Entity] = []
        content = chunk.content

        # Extract headings as concept entities
        for match in self.HEADING_PATTERN.finditer(content):
            level = len(match.group(1))
            name = match.group(2).strip()
            if len(name) > 3:
                entities.append(
                    Entity(
                        name=name,
                        entity_type="Concept" if level <= 2 else "SubConcept",
                        properties={"heading_level": level},
                        source_chunk_ids=[chunk.id],
                    )
                )

        # Extract specification/document references
        for match in self.SPEC_NUMBER_PATTERN.finditer(content):
            spec = match.group(1).strip()
            entities.append(
                Entity(
                    name=spec,
                    entity_type="Specification",
                    properties={"raw_match": spec},
                    source_chunk_ids=[chunk.id],
                )
            )

        # Extract part numbers
        for match in self.PART_NUMBER_PATTERN.finditer(content):
            pn = match.group(1).strip()
            entities.append(
                Entity(
                    name=pn,
                    entity_type="PartNumber",
                    properties={},
                    source_chunk_ids=[chunk.id],
                )
            )

        # Extract multi-word capitalized phrases (likely named concepts)
        for match in self.DEFINED_TERM_PATTERN.finditer(content):
            term = match.group(1).strip()
            if len(term) > 5 and len(term.split()) <= 5:
                entities.append(
                    Entity(
                        name=term,
                        entity_type="Term",
                        properties={},
                        source_chunk_ids=[chunk.id],
                    )
                )

        return entities

    def _extract_relations(
        self, chunk: Chunk, known_entities: dict[str, Entity]
    ) -> list[Relation]:
        """Extract relations from a chunk, linking known entities."""
        relations: list[Relation] = []

        for pattern, rel_type in self.RELATION_PATTERNS:
            for match in pattern.finditer(chunk.content):
                source_text = match.group(1).strip()[:100]
                target_text = match.group(2).strip()[:100]

                source_entity = self._resolve_entity(source_text, known_entities)
                target_entity = self._resolve_entity(target_text, known_entities)

                if source_entity and target_entity and source_entity.id != target_entity.id:
                    relations.append(
                        Relation(
                            source_id=source_entity.id,
                            target_id=target_entity.id,
                            relation_type=rel_type,
                            source_chunk_ids=[chunk.id],
                            confidence=0.7,
                            provenance="extracted",
                        )
                    )

        # Heading hierarchy relations (parent heading → child heading)
        headings = list(self.HEADING_PATTERN.finditer(chunk.content))
        for i in range(len(headings) - 1):
            parent_level = len(headings[i].group(1))
            child_level = len(headings[i + 1].group(1))
            if child_level > parent_level:
                parent_name = headings[i].group(2).strip()
                child_name = headings[i + 1].group(2).strip()
                parent = known_entities.get(parent_name)
                child = known_entities.get(child_name)
                if parent and child:
                    relations.append(
                        Relation(
                            source_id=parent.id,
                            target_id=child.id,
                            relation_type="HAS_SUBTOPIC",
                            source_chunk_ids=[chunk.id],
                            confidence=0.9,
                            provenance="inferred",
                        )
                    )

        return relations

    def _resolve_entity(
        self, text: str, known_entities: dict[str, Entity]
    ) -> Entity | None:
        """Try to match text to a known entity (exact or substring)."""
        if text in known_entities:
            return known_entities[text]

        # Try substring match
        for name, entity in known_entities.items():
            if name.lower() in text.lower() or text.lower() in name.lower():
                return entity

        return None


class CodeExtractor:
    """Extracts entities and relations from code AST metadata.

    Reads the ast_nodes attached by code_parser and produces Function,
    Class, Module entities with CALLS, IMPORTS, CONTAINS relations.
    """

    _ENTITY_NODE_TYPES = {
        "function_definition", "function_declaration", "method_definition",
        "arrow_function", "function_item",
    }
    _CLASS_NODE_TYPES = {
        "class_definition", "class_declaration", "class_specifier",
        "struct_item", "impl_item",
    }
    _IMPORT_NODE_TYPES = {
        "import_statement", "import_from_statement", "import_declaration",
        "use_declaration",
    }
    _CALL_NODE_TYPES = {"call_expression", "call"}

    def extract(self, chunks: list[Chunk]) -> ExtractionResult:
        """Extract code entities/relations from chunks with AST metadata."""
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []

        for chunk in chunks:
            ast_nodes = chunk.metadata.extra.get("ast_nodes", [])
            lang = chunk.metadata.extra.get("language", "unknown")
            source = chunk.metadata.source

            module_name = source.rsplit("/", 1)[-1] if "/" in source else source
            if module_name not in entities:
                mod_entity = Entity(
                    name=module_name,
                    entity_type="Module",
                    properties={"language": lang},
                    source_chunk_ids=[chunk.id],
                )
                entities[module_name] = mod_entity

            container_stack: list[str] = [module_name]

            for node in ast_nodes:
                ntype = node.get("type", "")
                name = node.get("name", "<anonymous>")
                if name == "<anonymous>":
                    continue

                if ntype in self._ENTITY_NODE_TYPES:
                    if name not in entities:
                        entities[name] = Entity(
                            name=name,
                            entity_type="Function",
                            properties={
                                "language": lang,
                                "start_line": node.get("start_line"),
                                "end_line": node.get("end_line"),
                            },
                            source_chunk_ids=[chunk.id],
                        )
                    parent = entities.get(container_stack[-1])
                    if parent:
                        relations.append(Relation(
                            source_id=parent.id,
                            target_id=entities[name].id,
                            relation_type="CONTAINS",
                            source_chunk_ids=[chunk.id],
                            provenance="extracted",
                        ))

                elif ntype in self._CLASS_NODE_TYPES:
                    if name not in entities:
                        entities[name] = Entity(
                            name=name,
                            entity_type="Class",
                            properties={
                                "language": lang,
                                "start_line": node.get("start_line"),
                                "end_line": node.get("end_line"),
                            },
                            source_chunk_ids=[chunk.id],
                        )
                    parent = entities.get(container_stack[-1])
                    if parent:
                        relations.append(Relation(
                            source_id=parent.id,
                            target_id=entities[name].id,
                            relation_type="CONTAINS",
                            source_chunk_ids=[chunk.id],
                            provenance="extracted",
                        ))
                    container_stack.append(name)

                elif ntype in self._IMPORT_NODE_TYPES:
                    import_name = name.replace("import ", "").replace("from ", "").split(" ")[0]
                    if import_name and import_name not in entities:
                        entities[import_name] = Entity(
                            name=import_name,
                            entity_type="Module",
                            properties={"language": lang, "imported": True},
                            source_chunk_ids=[chunk.id],
                        )
                    mod = entities.get(module_name)
                    imp = entities.get(import_name)
                    if mod and imp:
                        relations.append(Relation(
                            source_id=mod.id,
                            target_id=imp.id,
                            relation_type="IMPORTS",
                            source_chunk_ids=[chunk.id],
                            provenance="extracted",
                        ))

                elif ntype in self._CALL_NODE_TYPES:
                    if name not in entities:
                        entities[name] = Entity(
                            name=name,
                            entity_type="Function",
                            properties={"language": lang, "external": True},
                            source_chunk_ids=[chunk.id],
                        )
                    caller = entities.get(container_stack[-1])
                    callee = entities.get(name)
                    if caller and callee and caller.id != callee.id:
                        relations.append(Relation(
                            source_id=caller.id,
                            target_id=callee.id,
                            relation_type="CALLS",
                            source_chunk_ids=[chunk.id],
                            provenance="extracted",
                        ))

        logger.info("code_extraction_complete", entities=len(entities), relations=len(relations))
        return ExtractionResult(entities=list(entities.values()), relations=relations)
