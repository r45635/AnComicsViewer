"""LRU cache implementation for panel detection results.

Provides memory-efficient caching with configurable size limits.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar, Optional, List

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
