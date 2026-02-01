"""LRU cache implementation for panel detection results.

Provides memory-efficient caching with configurable size limits.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Generic, TypeVar, Optional, List, Callable
import time

from PySide6.QtCore import QRectF

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache with configurable maximum size.

    Automatically evicts least recently used items when capacity is exceeded.
    """

    def __init__(self, max_size: int = 50):
        """Initialize cache.

        Args:
            max_size: Maximum number of items to store
        """
        self._max_size = max_size
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = Lock()

    def get(self, key: K) -> Optional[V]:
        """Get item from cache, marking it as recently used.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self._cache:
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: K, value: V) -> None:
        """Add or update item in cache.

        Args:
            key: Cache key
            value: Value to store
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    # Evict oldest item
                    self._cache.popitem(last=False)
            self._cache[key] = value

    def remove(self, key: K) -> Optional[V]:
        """Remove item from cache.

        Args:
            key: Cache key

        Returns:
            Removed value or None if not found
        """
        with self._lock:
            return self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()

    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """Get number of items in cache."""
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        """Get maximum cache size."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        """Set maximum cache size, evicting if necessary."""
        with self._lock:
            self._max_size = max(1, value)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)


@dataclass
class CachedPanelResult:
    """Cached result from panel detection."""
    panels: List[QRectF]
    config_hash: int  # Hash of config used for detection

    @classmethod
    def empty(cls) -> "CachedPanelResult":
        return cls(panels=[], config_hash=0)


class PanelCache:
    """Specialized cache for panel detection results.

    Caches panels per page with config validation to ensure
    cache invalidation when detection parameters change.
    """

    def __init__(self, max_pages: int = 50):
        """Initialize panel cache.

        Args:
            max_pages: Maximum number of pages to cache
        """
        self._cache: LRUCache[int, CachedPanelResult] = LRUCache(max_pages)
        self._current_config_hash: int = 0

    def set_config_hash(self, config_hash: int) -> None:
        """Update current config hash.

        If changed, all cached entries become invalid.

        Args:
            config_hash: Hash of current detector configuration
        """
        if config_hash != self._current_config_hash:
            self._cache.clear()
            self._current_config_hash = config_hash

    def get(self, page: int) -> Optional[List[QRectF]]:
        """Get cached panels for a page.

        Args:
            page: Page number

        Returns:
            List of panel rects or None if not cached/invalid
        """
        result = self._cache.get(page)
        if result is None:
            return None
        if result.config_hash != self._current_config_hash:
            self._cache.remove(page)
            return None
        return result.panels

    def put(self, page: int, panels: List[QRectF]) -> None:
        """Cache panels for a page.

        Args:
            page: Page number
            panels: List of panel rects
        """
        self._cache.put(
            page,
            CachedPanelResult(panels=panels, config_hash=self._current_config_hash)
        )

    def invalidate_page(self, page: int) -> None:
        """Invalidate cache for a specific page.

        Args:
            page: Page number to invalidate
        """
        self._cache.remove(page)

    def clear(self) -> None:
        """Clear all cached panels."""
        self._cache.clear()

    def __contains__(self, page: int) -> bool:
        """Check if page is cached with valid config."""
        return self.get(page) is not None


@dataclass
class MemoryCacheEntry:
    """Cache entry with memory tracking."""
    data: any
    size_bytes: int
    timestamp: float = field(default_factory=time.time)


class MemoryAwareLRUCache(Generic[K, V]):
    """LRU cache with memory limit awareness.
    
    Tracks approximate memory usage and evicts when limit is reached.
    """
    
    def __init__(self, max_items: int = 100, max_memory_mb: float = 100.0):
        """Initialize cache.
        
        Args:
            max_items: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
        """
        self._max_items = max_items
        self._max_memory = int(max_memory_mb * 1024 * 1024)
        self._current_memory = 0
        self._cache: OrderedDict[K, MemoryCacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hit_count = 0
        self._miss_count = 0
    
    def get(self, key: K) -> Optional[V]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None
            self._cache.move_to_end(key)
            self._hit_count += 1
            return self._cache[key].data
    
    def put(self, key: K, value: V, size_bytes: int = 0) -> None:
        """Add or update item in cache with size tracking.
        
        Args:
            key: Cache key
            value: Value to store
            size_bytes: Estimated size in bytes (0 = auto-estimate)
        """
        with self._lock:
            # Estimate size if not provided
            if size_bytes == 0:
                size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._current_memory -= self._cache[key].size_bytes
                del self._cache[key]
            
            # Evict until under limits
            while (len(self._cache) >= self._max_items or 
                   self._current_memory + size_bytes > self._max_memory) and self._cache:
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._current_memory -= oldest_entry.size_bytes
            
            # Add new entry
            self._cache[key] = MemoryCacheEntry(data=value, size_bytes=size_bytes)
            self._current_memory += size_bytes
    
    def _estimate_size(self, value: any) -> int:
        """Estimate memory size of a value."""
        try:
            if isinstance(value, list):
                # For list of QRectF, each rect is ~32 bytes
                return len(value) * 32 + 64
            return sys.getsizeof(value)
        except Exception:
            return 256  # Default estimate
    
    def remove(self, key: K) -> Optional[V]:
        """Remove item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
            return entry.data
    
    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._cache
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
    
    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        with self._lock:
            return self._current_memory / (1024 * 1024)
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "items": len(self._cache),
                "max_items": self._max_items,
                "memory_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self._max_memory / (1024 * 1024),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": self.hit_rate,
            }
