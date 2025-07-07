# Fix Tokenizer Inefficiency

**Priority:** High
**Status:** Pending
**Commit SHA:** 360fbdf2661ddbc00ce2487da7e6e9a8d6684253

## Description

The `SimpleTokenizerV1` class has an inefficient `_int_to_str()` method that performs O(n) linear search through the vocabulary dictionary when converting integer indices back to token strings. This is unnecessary since the class already maintains an `int_to_str` dictionary for O(1) lookups.

## Justification

- **Performance Impact**: The current implementation causes unnecessary performance degradation during decoding operations
- **Code Duplication**: The method recreates functionality that already exists in the `int_to_str` instance variable
- **Maintainability**: Having two ways to do the same thing increases code complexity and potential for bugs
- **Scalability**: Linear search becomes increasingly expensive as vocabulary size grows

## Current State

In `src/llm_from_scratch/tokenizer.py:97-112`:

```python
def _int_to_str(self, index: int) -> str:
    """
    Convert integer index to corresponding token string.

    Args:
        index: Integer index to convert

    Returns:
        Token string corresponding to the index

    Raises:
        ValueError: If index is out of vocabulary range
    """
    if index < 0 or index >= len(self.vocabulary):
        raise ValueError(f"Index {index} out of range.")
    return list(self.vocabulary.keys())[list(self.vocabulary.values()).index(index)]
```

The class already has:
- `self.int_to_str: Dict[int, str]` created in `__init__` (line 59)
- Direct usage of `self.int_to_str[i]` in the `decode()` method (line 159)

## Proposed Changes

1. **Remove the inefficient method**: Delete the `_int_to_str()` method entirely
2. **Update validation**: Add bounds checking to the `decode()` method if needed
3. **Improve error handling**: Ensure proper error messages for out-of-bounds indices

## Task List

- [ ] Remove the `_int_to_str()` method from the `SimpleTokenizerV1` class
- [ ] Review all usages of `_int_to_str()` in the codebase (should be none)
- [ ] Add bounds checking to `decode()` method if not already present
- [ ] Update tests to verify proper error handling for invalid indices
- [ ] Run existing tests to ensure no regression
- [ ] Update type hints and docstrings as needed

## Design Patterns Applied

- **Don't Repeat Yourself (DRY)**: Eliminate code duplication
- **Single Responsibility**: Each data structure has one clear purpose
- **Performance Optimization**: Use appropriate data structures for the task

## Testing Requirements

- [ ] Verify existing tests still pass
- [ ] Add test for bounds checking with invalid indices
- [ ] Performance test to confirm O(1) decoding behavior
- [ ] Test edge cases (empty sequences, single tokens)

## Breaking Changes

**None** - This is an internal method removal that doesn't affect the public API.

## Implementation Notes

The `decode()` method already uses `self.int_to_str[i]` directly, so the inefficient method appears to be unused. This makes the refactoring even safer as it's purely removing dead code.

## Success Criteria

- [ ] All existing tests pass
- [ ] No performance regression in decoding
- [ ] Cleaner, more maintainable code
- [ ] Proper error handling for edge cases
