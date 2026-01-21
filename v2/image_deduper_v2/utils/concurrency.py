"""Concurrency utilities for parallel processing.

This module provides helpers for running tasks in thread pools
with proper exception handling and cancellation support.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Generator, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_in_executor(
    func: Callable[..., R],
    items: Iterable[T],
    max_workers: int = 8,
    item_to_args: Callable[[T], tuple[Any, ...]] | None = None,
    item_to_kwargs: Callable[[T], dict[str, Any]] | None = None,
) -> Generator[tuple[T, R | None, Exception | None], None, None]:
    """Execute a function over items using a thread pool.
    
    Yields results as they complete, allowing for streaming processing.
    
    Args:
        func: Function to execute for each item.
        items: Iterable of items to process.
        max_workers: Maximum number of concurrent workers.
        item_to_args: Optional function to convert item to positional args.
        item_to_kwargs: Optional function to convert item to keyword args.
        
    Yields:
        Tuples of (item, result, exception) where either result or exception
        is None.
        
    Example:
        >>> def process(path: Path) -> int:
        ...     return path.stat().st_size
        >>> 
        >>> for path, size, error in run_in_executor(process, paths):
        ...     if error:
        ...         print(f"Error: {path}: {error}")
        ...     else:
        ...         print(f"{path}: {size} bytes")
    """
    items_list = list(items)
    if not items_list:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        if item_to_args:
            futures: dict[Future[R], T] = {
                executor.submit(func, *item_to_args(item)): item
                for item in items_list
            }
        elif item_to_kwargs:
            futures = {
                executor.submit(func, **item_to_kwargs(item)): item
                for item in items_list
            }
        else:
            futures = {
                executor.submit(func, item): item
                for item in items_list
            }

        # Yield results as they complete
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                yield (item, result, None)
            except Exception as e:
                yield (item, None, e)


async def run_in_executor_async(
    func: Callable[..., R],
    *args: Any,
    executor: ThreadPoolExecutor | None = None,
    **kwargs: Any,
) -> R:
    """Run a blocking function in an executor asynchronously.
    
    Args:
        func: Function to execute.
        *args: Positional arguments for the function.
        executor: Optional executor to use (default: asyncio default).
        **kwargs: Keyword arguments for the function.
        
    Returns:
        Result of the function.
        
    Example:
        >>> async def main():
        ...     result = await run_in_executor_async(
        ...         expensive_operation, arg1, arg2
        ...     )
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(executor, func, *args)


def chunk_iterable(
    iterable: Iterable[T],
    chunk_size: int,
) -> Generator[list[T], None, None]:
    """Split an iterable into chunks of a specified size.
    
    Args:
        iterable: The iterable to chunk.
        chunk_size: Maximum size of each chunk.
        
    Yields:
        Lists of items, each with at most chunk_size elements.
        
    Example:
        >>> items = range(10)
        >>> for chunk in chunk_iterable(items, 3):
        ...     print(chunk)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    chunk: list[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class TaskGroup:
    """A group of concurrent tasks with collective management.
    
    Provides a context manager for running multiple tasks and
    collecting their results.
    
    Example:
        >>> with TaskGroup(max_workers=4) as group:
        ...     group.submit(process, item1)
        ...     group.submit(process, item2)
        ...     results = group.get_results()
    """

    def __init__(self, max_workers: int = 8) -> None:
        """Initialize the task group.
        
        Args:
            max_workers: Maximum number of concurrent workers.
        """
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[Any]] = []

    def __enter__(self) -> "TaskGroup":
        """Enter the context manager."""
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        """Submit a task to the group.
        
        Args:
            fn: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
            
        Returns:
            Future representing the pending result.
            
        Raises:
            RuntimeError: If called outside of context manager.
        """
        if self._executor is None:
            raise RuntimeError("TaskGroup must be used as a context manager")
        
        if kwargs:
            fn = partial(fn, **kwargs)
        
        future = self._executor.submit(fn, *args)
        self._futures.append(future)
        return future

    def get_results(self, raise_exceptions: bool = False) -> list[tuple[Any, Exception | None]]:
        """Collect all results from submitted tasks.
        
        Args:
            raise_exceptions: If True, raise the first exception encountered.
            
        Returns:
            List of (result, exception) tuples. If successful, exception is None;
            if failed, result is None.
        """
        results: list[tuple[Any, Exception | None]] = []
        for future in self._futures:
            try:
                result = future.result()
                results.append((result, None))
            except Exception as e:
                if raise_exceptions:
                    raise
                results.append((None, e))
        return results
