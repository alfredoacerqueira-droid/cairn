"""Structural retriever — exact block-identity + reference matching.

This retriever matches query terms against block IDs and internal references.
Unlike embeddings, it handles exact structural concepts (resource types,
variable names, module references) that embeddings conflate. Offline, pure
string-based, deterministic.

ID format: filepath:block_name:line_start
Block names use dot-notation (e.g., resource.aws_iam_role.this) or class
notation (e.g., ClassName.method_name).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _tokenize_block_name(name: str) -> set[str]:
    """Tokenize a block name into individual terms and subtypes.

    Examples:
        'resource.aws_iam_role.this' -> {'resource', 'aws', 'iam', 'role',
                                          'this', 'aws_iam_role', 'iam_role'}
        'variable.cluster_version' -> {'variable', 'cluster', 'version',
                                        'cluster_version'}
        'ClassName.method_name' -> {'classname', 'method', 'name',
                                     'method_name'}
    """
    tokens: set[str] = set()
    # Split on dot, hyphen, underscore
    parts = re.split(r"[._-]+", name.lower())
    tokens.update(p for p in parts if p)

    # Also add underscore-joined subparts (e.g., 'aws_iam_role')
    underscored = re.split(r"[.-]+", name.lower())
    underscored_joined = "_".join(p for p in underscored if p)
    if underscored_joined:
        tokens.add(underscored_joined)

    # Add hyphen-joined subparts
    hyphenated = re.split(r"[_.]", name.lower())
    hyphenated_joined = "-".join(p for p in hyphenated if p)
    if hyphenated_joined and hyphenated_joined != underscored_joined:
        tokens.add(hyphenated_joined)

    return tokens


def _extract_identifiers(text: str) -> set[str]:
    """Extract identifiers and references from block text.

    Matches patterns like:
        - var.cluster_version
        - module.eks
        - aws_iam_role.this.arn
        - kind: Deployment
        - metadata.name
        - resource "aws_iam_role" "name" (Terraform resource declarations)
        - data "aws_availability_zones" (Terraform data declarations)

    Returns lowercased, deduplicated identifier names.
    """
    identifiers: set[str] = set()

    # Pattern: Terraform resource/data declarations: resource "TYPE" "name"
    pattern_tf = r'(?:resource|data)\s+"([a-z_][a-z0-9_]*)"\s+"([a-z_][a-z0-9_-]*)"'
    for match in re.finditer(pattern_tf, text.lower()):
        res_type = match.group(1)
        res_name = match.group(2)
        identifiers.add(res_type)
        identifiers.add(res_name)
        identifiers.add(f"{res_type}.{res_name}")

    # Pattern: resource_type.instance (e.g., aws_iam_role.this)
    pattern1 = r"([a-z_][a-z0-9_]*)\s*\.\s*([a-z_][a-z0-9_-]*)"
    for match in re.finditer(pattern1, text.lower()):
        identifiers.add(match.group(1))
        identifiers.add(match.group(2))
        identifiers.add(f"{match.group(1)}.{match.group(2)}")

    # Pattern: var.X, module.X, data.X, resource.X (explicit prefixes)
    pattern2 = r"\b(var|module|data|resource|output|kind|metadata)[\s.]+([a-z_][a-z0-9_-]*)"
    for match in re.finditer(pattern2, text.lower()):
        identifiers.add(match.group(2))
        identifiers.add(f"{match.group(1)}.{match.group(2)}")

    # Pattern: Kind: X (Kubernetes-style, case-insensitive)
    pattern3 = r"\bkind\s*:\s*([A-Za-z][A-Za-z0-9-]*)"
    for match in re.finditer(pattern3, text, re.IGNORECASE):
        identifiers.add(match.group(1).lower())

    return identifiers


class StructuralRetriever:
    """Block-identity and reference-based retrieval."""

    def __init__(self):
        # Map of token -> list of doc_ids containing that token
        self._token_index: dict[str, list[str]] = defaultdict(list)
        # Map of referenced identifier -> list of doc_ids that reference it
        self._reference_index: dict[str, list[str]] = defaultdict(list)
        # Store the actual block names and text for scoring
        self._blocks: dict[str, dict[str, Any]] = {}

    def index(self, items: list[dict[str, Any]]) -> None:
        """Index documents from a list of dicts with 'id' and 'text' keys.

        ID format: filepath:block_name:line_start
        """
        self._token_index = defaultdict(list)
        self._reference_index = defaultdict(list)
        self._blocks = {}

        for item in items:
            doc_id = item["id"]
            text = item.get("text", "")

            # Parse ID: filepath:block_name:line_start
            # block_name may contain dots
            parts = doc_id.rsplit(":", 2)
            if len(parts) < 2:
                continue
            block_name = parts[1]

            # Index block-name tokens
            name_tokens = _tokenize_block_name(block_name)
            for token in name_tokens:
                self._token_index[token].append(doc_id)

            # Index references within the block text
            references = _extract_identifiers(text)
            for ref in references:
                self._reference_index[ref].append(doc_id)

            # Store block metadata for scoring
            self._blocks[doc_id] = {
                "block_name": block_name,
                "text": text,
                "name_tokens": name_tokens,
            }

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Score and rank documents by structural match.

        Scoring strategy:
        1. Extract query terms (lowercased identifiers).
        2. For each block, compute:
           - name_match: overlap between query terms and block name tokens
           - reference_match: block's references that appear in query
           - substring_match: block type substring overlap
        3. Combine scores and return top_k.
        """
        if not self._blocks:
            return []

        query_lower = query.lower()
        query_terms = set(re.findall(r"[a-z0-9]+", query_lower))
        query_identifiers = _extract_identifiers(query)

        scores: list[tuple[float, str, str]] = []

        for doc_id, block_info in self._blocks.items():
            block_name = block_info["block_name"]
            text = block_info["text"]
            name_tokens = block_info["name_tokens"]

            score = 0.0

            # 1. Block-name token overlap (high weight)
            name_overlap = len(name_tokens & query_terms)
            if name_overlap > 0:
                score += 100.0 * name_overlap

            # 2. Query identifier found as reference in block (medium)
            block_refs = _extract_identifiers(text)
            ref_overlap = len(query_identifiers & block_refs)
            if ref_overlap > 0:
                score += 30.0 * ref_overlap

            # 3. Query TERMS found in block's extracted identifiers
            # (this allows "vpc peering" to match "aws_vpc_peering...")
            block_ref_terms = set()
            for ref in block_refs:
                block_ref_terms.update(re.findall(r"[a-z0-9]+", ref.lower()))
            term_overlap = len(query_terms & block_ref_terms)
            if term_overlap > 0:
                score += 25.0 * term_overlap

            # 4. Substring matching: query terms in block type
            block_type_parts = re.split(r"[._-]+", block_name.lower())
            for query_term in query_terms:
                for part in block_type_parts:
                    if query_term in part and query_term != part:
                        score += 5.0

            # 5. Bonus if block type contains all query parts
            if len(query_terms) >= 2:
                query_pair = sorted(query_terms)
                block_pair = sorted(block_type_parts)
                if all(qt in block_pair for qt in query_pair):
                    score += 15.0

            if score > 0:
                scores.append((score, doc_id, text))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, doc_id, text in scores[:top_k]:
            results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "score": round(score, 4),
                    "source": "structural",
                }
            )
        return results
