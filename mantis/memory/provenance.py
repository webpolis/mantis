"""
Evidence trust by provenance, shared by both memory tiers.

Entries below the engine's `min_evidence_trust` are never cited as facts, so
a generated claim gains no evidential status by being stored.
"""

from typing import Dict

TRUST = {'user': 2, 'external': 2, 'stage2': 2, 'document': 2, 'model_verified': 1, 'interaction': 0, 'model': 0}


def trust_level(metadata: Dict) -> int:
    if 'trust' in metadata:
        return int(metadata['trust'])
    return TRUST.get(metadata.get('source', 'model'), 0)
