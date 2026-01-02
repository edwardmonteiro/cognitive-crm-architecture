"""
Identity Resolution - Entity Linking Across Systems

From Section 4.1: "Identity resolution: entity linking across systems
(account/contact/opportunity)"

This module provides:
- Cross-system entity matching
- Deduplication
- Hierarchical entity relationships
- Confidence scoring for matches

Identity resolution is critical for:
- Accurate customer 360 views
- Correct attribution of interactions
- Reliable sensemaking across channels
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import UUID, uuid4


class EntityType(Enum):
    """Types of entities in the CRM ecosystem."""
    ACCOUNT = "account"
    CONTACT = "contact"
    OPPORTUNITY = "opportunity"
    INTERACTION = "interaction"


class MatchConfidence(Enum):
    """Confidence levels for entity matches."""
    EXACT = "exact"           # Exact ID match
    HIGH = "high"             # Multiple strong signals
    MEDIUM = "medium"         # Some signals match
    LOW = "low"               # Weak match, needs review
    NO_MATCH = "no_match"


@dataclass
class EntityReference:
    """
    Reference to an entity across systems.

    Captures:
    - Source system identifier
    - External IDs
    - Matching attributes
    """
    id: UUID = field(default_factory=uuid4)
    entity_type: EntityType = EntityType.ACCOUNT
    source_system: str = ""
    source_id: str = ""

    # Core identifying attributes
    canonical_id: Optional[UUID] = None  # Resolved canonical entity
    attributes: dict = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_seen_at: datetime = field(default_factory=datetime.now)


@dataclass
class MatchResult:
    """Result of an identity resolution attempt."""
    source_ref: EntityReference = None
    matched_ref: Optional[EntityReference] = None
    confidence: MatchConfidence = MatchConfidence.NO_MATCH
    match_score: float = 0.0
    match_reasons: list = field(default_factory=list)
    requires_review: bool = False


class MatchingRule:
    """
    Base class for entity matching rules.

    Rules define how to match entities based on attributes.
    Multiple rules can be combined for comprehensive matching.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def match(self, ref1: EntityReference, ref2: EntityReference) -> tuple[float, str]:
        """
        Attempt to match two entity references.
        Returns (score, reason) where score is 0.0 to 1.0.
        """
        raise NotImplementedError


class ExactIdMatch(MatchingRule):
    """Match on exact external ID."""

    def __init__(self, id_field: str, weight: float = 1.0):
        super().__init__(weight)
        self.id_field = id_field

    def match(self, ref1: EntityReference, ref2: EntityReference) -> tuple[float, str]:
        val1 = ref1.attributes.get(self.id_field)
        val2 = ref2.attributes.get(self.id_field)

        if val1 and val2 and val1 == val2:
            return (1.0, f"Exact match on {self.id_field}")
        return (0.0, "")


class EmailDomainMatch(MatchingRule):
    """Match accounts by email domain."""

    def match(self, ref1: EntityReference, ref2: EntityReference) -> tuple[float, str]:
        domain1 = ref1.attributes.get("email_domain")
        domain2 = ref2.attributes.get("email_domain")

        if domain1 and domain2:
            if domain1.lower() == domain2.lower():
                return (0.9, "Email domain match")
        return (0.0, "")


class NameSimilarityMatch(MatchingRule):
    """Match on name similarity using fuzzy matching."""

    def __init__(self, name_field: str = "name", threshold: float = 0.85, weight: float = 0.8):
        super().__init__(weight)
        self.name_field = name_field
        self.threshold = threshold

    def match(self, ref1: EntityReference, ref2: EntityReference) -> tuple[float, str]:
        name1 = ref1.attributes.get(self.name_field, "")
        name2 = ref2.attributes.get(self.name_field, "")

        if not name1 or not name2:
            return (0.0, "")

        similarity = self._calculate_similarity(name1, name2)

        if similarity >= self.threshold:
            return (similarity, f"Name similarity: {similarity:.2f}")
        return (0.0, "")

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity."""
        s1, s2 = s1.lower(), s2.lower()

        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Simple Levenshtein distance
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,
                    matrix[i][j-1] + 1,
                    matrix[i-1][j-1] + cost
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)

        return 1.0 - (distance / max_len)


class IdentityResolver:
    """
    Main identity resolution engine.

    Coordinates matching rules to resolve entities across systems
    into canonical identities.
    """

    def __init__(self):
        self._rules: dict[EntityType, list[MatchingRule]] = {
            EntityType.ACCOUNT: [
                ExactIdMatch("domain", weight=1.0),
                EmailDomainMatch(weight=0.9),
                NameSimilarityMatch("name", weight=0.8)
            ],
            EntityType.CONTACT: [
                ExactIdMatch("email", weight=1.0),
                NameSimilarityMatch("full_name", weight=0.7)
            ],
            EntityType.OPPORTUNITY: [
                ExactIdMatch("external_id", weight=1.0),
                ExactIdMatch("deal_id", weight=1.0)
            ]
        }
        self._entity_store: dict[str, list[EntityReference]] = {}
        self._canonical_map: dict[UUID, UUID] = {}  # source_id -> canonical_id

    def add_rule(self, entity_type: EntityType, rule: MatchingRule) -> None:
        """Add a matching rule for an entity type."""
        if entity_type not in self._rules:
            self._rules[entity_type] = []
        self._rules[entity_type].append(rule)

    def register_entity(self, ref: EntityReference) -> MatchResult:
        """
        Register an entity reference and attempt to resolve identity.

        Returns the match result with the canonical entity if found.
        """
        entity_key = f"{ref.entity_type.value}:{ref.source_system}"

        if entity_key not in self._entity_store:
            self._entity_store[entity_key] = []

        # Try to find existing match
        best_match = None
        best_score = 0.0
        match_reasons = []

        rules = self._rules.get(ref.entity_type, [])

        for existing_ref in self._entity_store[entity_key]:
            total_score = 0.0
            total_weight = 0.0
            reasons = []

            for rule in rules:
                score, reason = rule.match(ref, existing_ref)
                if score > 0:
                    total_score += score * rule.weight
                    total_weight += rule.weight
                    reasons.append(reason)

            if total_weight > 0:
                normalized_score = total_score / total_weight
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = existing_ref
                    match_reasons = reasons

        # Determine confidence level
        if best_score >= 0.95:
            confidence = MatchConfidence.EXACT
        elif best_score >= 0.8:
            confidence = MatchConfidence.HIGH
        elif best_score >= 0.6:
            confidence = MatchConfidence.MEDIUM
        elif best_score >= 0.4:
            confidence = MatchConfidence.LOW
        else:
            confidence = MatchConfidence.NO_MATCH
            best_match = None

        # Create new canonical if no match
        if best_match is None:
            ref.canonical_id = uuid4()
            self._entity_store[entity_key].append(ref)
        else:
            ref.canonical_id = best_match.canonical_id

        self._canonical_map[ref.id] = ref.canonical_id

        return MatchResult(
            source_ref=ref,
            matched_ref=best_match,
            confidence=confidence,
            match_score=best_score,
            match_reasons=match_reasons,
            requires_review=confidence in [MatchConfidence.LOW, MatchConfidence.MEDIUM]
        )

    def get_canonical_id(self, source_id: UUID) -> Optional[UUID]:
        """Get canonical ID for a source entity."""
        return self._canonical_map.get(source_id)

    def get_all_references(self, canonical_id: UUID) -> list[EntityReference]:
        """Get all entity references for a canonical ID."""
        results = []
        for refs in self._entity_store.values():
            for ref in refs:
                if ref.canonical_id == canonical_id:
                    results.append(ref)
        return results


@dataclass
class EntityGraph:
    """
    Graph of entity relationships.

    Captures:
    - Account -> Contact relationships
    - Account -> Opportunity relationships
    - Contact <-> Opportunity associations

    Enables traversal for context building.
    """
    _edges: dict = field(default_factory=dict)

    def add_edge(
        self,
        from_id: UUID,
        from_type: EntityType,
        to_id: UUID,
        to_type: EntityType,
        relationship: str
    ) -> None:
        """Add a relationship edge."""
        key = str(from_id)
        if key not in self._edges:
            self._edges[key] = []

        self._edges[key].append({
            "from_id": from_id,
            "from_type": from_type,
            "to_id": to_id,
            "to_type": to_type,
            "relationship": relationship
        })

    def get_related(
        self,
        entity_id: UUID,
        relationship: str = None,
        target_type: EntityType = None
    ) -> list[dict]:
        """Get related entities."""
        key = str(entity_id)
        edges = self._edges.get(key, [])

        if relationship:
            edges = [e for e in edges if e["relationship"] == relationship]
        if target_type:
            edges = [e for e in edges if e["to_type"] == target_type]

        return edges

    def get_account_context(self, account_id: UUID) -> dict:
        """Get full context for an account."""
        contacts = self.get_related(
            account_id,
            relationship="has_contact",
            target_type=EntityType.CONTACT
        )
        opportunities = self.get_related(
            account_id,
            relationship="has_opportunity",
            target_type=EntityType.OPPORTUNITY
        )

        return {
            "account_id": account_id,
            "contact_ids": [e["to_id"] for e in contacts],
            "opportunity_ids": [e["to_id"] for e in opportunities]
        }
