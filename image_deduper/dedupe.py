from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional


def hamming_distance(a: int, b: int) -> int:
    """Computes Hamming distance between two integers.

    Args:
        a: First integer bitset.
        b: Second integer bitset.

    Returns:
        Number of differing bits.
    """
    return (a ^ b).bit_count()


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculates distance in meters between two GPS coordinates.

    Args:
        lat1: Latitude of first point.
        lon1: Longitude of first point.
        lat2: Latitude of second point.
        lon2: Longitude of second point.

    Returns:
        Distance in meters.
    """
    r = 6371000
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


@dataclass
class UnionFind:
    """Union-Find (Disjoint Set Union) data structure."""

    parent: List[int]
    rank: List[int]

    @classmethod
    def create(cls, n: int) -> "UnionFind":
        """Creates a UnionFind structure with n elements.

        Args:
            n: Number of elements.

        Returns:
            UnionFind instance.
        """
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        """Finds the representative of x with path compression.

        Args:
            x: Element index.

        Returns:
            Representative index.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Unions two sets.

        Args:
            a: First element index.
            b: Second element index.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
            return
        if self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
            return
        self.parent[rb] = ra
        self.rank[ra] += 1


def lsh_keys(phash: int, bits: int, bands: int) -> List[int]:
    """Creates locality-sensitive keys by splitting a bitset into bands.

    Args:
        phash: Hash bitset.
        bits: Total bits in phash.
        bands: Number of bands.

    Returns:
        List of integer keys for each band.
    """
    band_bits = max(1, bits // max(1, bands))
    keys = []
    for i in range(bands):
        shift = i * band_bits
        mask = (1 << band_bits) - 1
        keys.append((phash >> shift) & mask)
    return keys


def cluster_near_duplicates(phashes: List[int], threshold: int, total_bits: int) -> List[List[int]]:
    """Clusters near-duplicate images using LSH candidate generation + union-find.

    Args:
        phashes: List of perceptual hashes.
        threshold: Maximum Hamming distance to cluster.
        total_bits: Total phash bits.

    Returns:
        List of clusters, each containing indices into phashes.
    """
    n = len(phashes)
    if n == 0:
        return []
    bands = 4 if total_bits >= 64 else 2
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, h in enumerate(phashes):
        for band_idx, key in enumerate(lsh_keys(h, total_bits, bands)):
            buckets.setdefault((band_idx, key), []).append(idx)

    uf = UnionFind.create(n)
    seen_pairs: set[Tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        m = sorted(members)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                a, b = m[i], m[j]
                pair = (a, b)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if hamming_distance(phashes[a], phashes[b]) <= threshold:
                    uf.union(a, b)

    clusters_map: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        clusters_map.setdefault(r, []).append(i)
    return list(clusters_map.values())


def cluster_by_gps_location(
    coordinates: List[Tuple[Optional[float], Optional[float]]],
    distance_threshold: float,
) -> List[List[int]]:
    """Clusters images by GPS location using distance threshold.

    Args:
        coordinates: List of (latitude, longitude) tuples. None values allowed.
        distance_threshold: Maximum distance in meters to cluster.

    Returns:
        List of clusters, each containing indices.
    """
    n = len(coordinates)
    if n == 0:
        return []

    uf = UnionFind.create(n)

    gps_indices = [
        i for i, (lat, lon) in enumerate(coordinates) 
        if lat is not None and lon is not None
    ]

    for i in range(len(gps_indices)):
        for j in range(i + 1, len(gps_indices)):
            idx_a = gps_indices[i]
            idx_b = gps_indices[j]
            lat1, lon1 = coordinates[idx_a]
            lat2, lon2 = coordinates[idx_b]

            dist = haversine_distance(lat1, lon1, lat2, lon2)
            if dist <= distance_threshold:
                uf.union(idx_a, idx_b)

    clusters_map: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        clusters_map.setdefault(r, []).append(i)
    return list(clusters_map.values())
