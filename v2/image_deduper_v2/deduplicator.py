"""Deduplication strategies module.

This module provides various strategies for detecting and
removing duplicate images based on different criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from image_deduper_v2.hasher import hamming_distance
from image_deduper_v2.models import ImageRecord
from image_deduper_v2.protocols import DeduplicationStrategy
from image_deduper_v2.utils.gps import haversine_distance
from image_deduper_v2.utils.logging import get_logger


@dataclass
class UnionFind:
    """Union-Find (Disjoint Set Union) data structure.
    
    Efficiently tracks connected components for clustering.
    
    Attributes:
        parent: List mapping each element to its parent.
        rank: List tracking tree depth for union by rank.
    """
    
    parent: List[int]
    rank: List[int]

    @classmethod
    def create(cls, n: int) -> "UnionFind":
        """Create a UnionFind structure with n elements.
        
        Args:
            n: Number of elements.
            
        Returns:
            New UnionFind instance.
        """
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        """Find the representative of x with path compression.
        
        Args:
            x: Element index.
            
        Returns:
            Representative (root) index.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Union two sets by rank.
        
        Args:
            a: First element index.
            b: Second element index.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def get_clusters(self) -> List[List[int]]:
        """Get all clusters as lists of indices.
        
        Returns:
            List of clusters, each containing member indices.
        """
        clusters_map: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            clusters_map.setdefault(root, []).append(i)
        return list(clusters_map.values())


def lsh_keys(phash: int, total_bits: int, bands: int) -> List[int]:
    """Create LSH keys by splitting a hash into bands.
    
    Locality-Sensitive Hashing divides the hash into bands
    to create a candidate generation scheme.
    
    Args:
        phash: Hash value as integer.
        total_bits: Total bits in the hash.
        bands: Number of bands to create.
        
    Returns:
        List of integer keys, one per band.
    """
    band_bits = max(1, total_bits // max(1, bands))
    keys = []
    for i in range(bands):
        shift = i * band_bits
        mask = (1 << band_bits) - 1
        keys.append((phash >> shift) & mask)
    return keys


class ExactHashDeduplicator(DeduplicationStrategy):
    """Deduplication strategy using exact SHA-256 hash matching.
    
    Groups images with identical file contents.
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "ExactHash"

    def find_duplicates(
        self,
        records: Sequence[ImageRecord],
    ) -> List[List[int]]:
        """Find groups of exact duplicate images.
        
        Args:
            records: Sequence of image records.
            
        Returns:
            List of clusters containing duplicate indices.
        """
        if not records:
            return []

        # Group by SHA-256 hash
        by_hash: Dict[str, List[int]] = {}
        for idx, record in enumerate(records):
            by_hash.setdefault(record.sha256, []).append(idx)

        return list(by_hash.values())


class PerceptualHashDeduplicator(DeduplicationStrategy):
    """Deduplication strategy using perceptual hash similarity.
    
    Uses LSH for candidate generation and Hamming distance
    for similarity comparison.
    
    Attributes:
        threshold: Maximum Hamming distance for duplicates.
        total_bits: Total bits in the perceptual hash.
    """

    def __init__(
        self,
        threshold: int = 10,
        total_bits: int = 256,
    ) -> None:
        """Initialize the deduplicator.
        
        Args:
            threshold: Maximum Hamming distance to consider similar.
            total_bits: Total bits in the hash (hash_size squared).
        """
        self._threshold = threshold
        self._total_bits = total_bits
        self._logger = get_logger("deduplicator.phash")

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "PerceptualHash"

    @property
    def threshold(self) -> int:
        """Return the Hamming distance threshold."""
        return self._threshold

    def find_duplicates(
        self,
        records: Sequence[ImageRecord],
    ) -> List[List[int]]:
        """Find groups of perceptually similar images.
        
        Uses LSH for efficient candidate generation followed by
        exact Hamming distance computation.
        
        Args:
            records: Sequence of image records.
            
        Returns:
            List of clusters containing similar image indices.
        """
        n = len(records)
        if n == 0:
            return []

        # Determine number of LSH bands
        bands = 4 if self._total_bits >= 64 else 2

        # Build LSH buckets
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for idx, record in enumerate(records):
            for band_idx, key in enumerate(lsh_keys(record.phash, self._total_bits, bands)):
                buckets.setdefault((band_idx, key), []).append(idx)

        # Create union-find for clustering
        uf = UnionFind.create(n)
        seen_pairs: set[Tuple[int, int]] = set()

        # Check candidates
        for members in buckets.values():
            if len(members) < 2:
                continue
            
            # Check all pairs within bucket
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    pair = (min(a, b), max(a, b))
                    
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    
                    # Compute actual Hamming distance
                    dist = hamming_distance(records[a].phash, records[b].phash)
                    if dist <= self._threshold:
                        uf.union(a, b)

        return uf.get_clusters()


class GPSProximityDeduplicator(DeduplicationStrategy):
    """Deduplication strategy based on GPS location proximity.
    
    Groups images taken within a specified distance of each other.
    
    Attributes:
        distance_threshold: Maximum distance in meters.
    """

    def __init__(self, distance_threshold: float = 50.0) -> None:
        """Initialize the deduplicator.
        
        Args:
            distance_threshold: Maximum distance in meters to group.
        """
        self._distance_threshold = distance_threshold
        self._logger = get_logger("deduplicator.gps")

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "GPSProximity"

    @property
    def distance_threshold(self) -> float:
        """Return the distance threshold in meters."""
        return self._distance_threshold

    def find_duplicates(
        self,
        records: Sequence[ImageRecord],
    ) -> List[List[int]]:
        """Find groups of images at similar GPS locations.
        
        Note: Images without GPS data are placed in singleton clusters.
        
        Args:
            records: Sequence of image records.
            
        Returns:
            List of clusters containing nearby image indices.
        """
        n = len(records)
        if n == 0:
            return []

        uf = UnionFind.create(n)

        # Get indices of images with GPS
        gps_indices = [i for i, r in enumerate(records) if r.has_gps]

        # Compare all pairs with GPS (O(n²) but typically n is small)
        for i in range(len(gps_indices)):
            for j in range(i + 1, len(gps_indices)):
                idx_a = gps_indices[i]
                idx_b = gps_indices[j]
                
                rec_a = records[idx_a]
                rec_b = records[idx_b]
                
                # Both have GPS (already verified)
                distance = rec_a.gps.distance_to(rec_b.gps)  # type: ignore
                
                if distance <= self._distance_threshold:
                    uf.union(idx_a, idx_b)

        return uf.get_clusters()


class CompositeDeduplicator:
    """Combines multiple deduplication strategies.
    
    Applies strategies in sequence, selecting the best image
    from each group after each strategy.
    """

    def __init__(
        self,
        strategies: List[DeduplicationStrategy],
        prefer_gps: bool = True,
    ) -> None:
        """Initialize the composite deduplicator.
        
        Args:
            strategies: List of strategies to apply in order.
            prefer_gps: Whether to prefer images with GPS data.
        """
        self._strategies = strategies
        self._prefer_gps = prefer_gps
        self._logger = get_logger("deduplicator.composite")

    def deduplicate(
        self,
        records: List[ImageRecord],
    ) -> Tuple[List[ImageRecord], Dict[str, int]]:
        """Apply all strategies and return deduplicated records.
        
        Args:
            records: List of image records.
            
        Returns:
            Tuple of (deduplicated records, counts per strategy).
        """
        removed_counts: Dict[str, int] = {}
        current = records

        for strategy in self._strategies:
            self._logger.info(f"Applying {strategy.name} deduplication...")
            
            clusters = strategy.find_duplicates(current)
            
            # Select best from each cluster
            selected: List[ImageRecord] = []
            removed = 0
            
            for cluster in clusters:
                group = [current[i] for i in cluster]
                best = self._select_best(group)
                selected.append(best)
                removed += len(group) - 1
            
            removed_counts[strategy.name] = removed
            current = selected
            
            self._logger.info(f"  Removed {removed} duplicates, {len(current)} remaining")

        return current, removed_counts

    def _select_best(self, group: List[ImageRecord]) -> ImageRecord:
        """Select the best image from a group of duplicates.
        
        Args:
            group: List of duplicate image records.
            
        Returns:
            The best image record from the group.
        """
        return max(
            group,
            key=lambda r: (
                r.has_gps if self._prefer_gps else False,
                r.metrics.quality_score,
                r.metrics.resolution,
                -r.metrics.file_size,  # Prefer smaller files
            ),
        )
