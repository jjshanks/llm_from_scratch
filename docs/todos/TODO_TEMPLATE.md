# [TODO Title]

**Priority:** [High/Medium/Low]
**Status:** [Pending/In Progress/Completed/On Hold]
**Commit SHA:** [git commit hash this TODO is based on]

## Description

[Clear, concise description of what needs to be done. 1-2 paragraphs explaining the task or refactoring opportunity.]

## Justification

[Why this work is needed. Include bullets covering:]
- **[Benefit Category]**: [Specific benefit explanation]
- **[Another Category]**: [Another benefit]
- **[Performance/Maintainability/User Experience/etc.]**: [Impact description]

## Current State

[Description of how things work currently. Include code examples where relevant:]

```python
# Current implementation example
def current_function():
    # Existing code that needs refactoring
    pass
```

[Explain limitations or problems with current approach]

## Proposed Changes

[Detailed technical approach for implementing the solution:]

1. **[Major Change 1]**: [Description]
2. **[Major Change 2]**: [Description]
3. **[Major Change 3]**: [Description]

## Task List

- [ ] [Specific implementation task 1]
- [ ] [Specific implementation task 2]
  - [ ] [Sub-task if needed]
  - [ ] [Another sub-task]
- [ ] [Testing task]
- [ ] [Documentation task]
- [ ] [Integration task]

## Design Patterns Applied

- **[Pattern Name]**: [How this pattern applies to the solution]
- **[Another Pattern]**: [Explanation of usage]
- **[Third Pattern]**: [Benefits in this context]

## [Optional: Enhanced/Proposed Structure Section]

[If relevant, include detailed code structure or class definitions]

```python
class ExampleClass:
    """Example of proposed implementation."""

    def __init__(self, param: str) -> None:
        """Initialize with parameters."""
        pass

    def method(self) -> str:
        """Example method implementation."""
        pass
```

## Testing Requirements

- [ ] [Specific test case 1]
- [ ] [Specific test case 2]
- [ ] [Edge case testing]
- [ ] [Integration testing]
- [ ] [Performance testing if relevant]
- [ ] [Backwards compatibility testing]

## Breaking Changes

**[None/Minor/Major]** - [Explanation of any breaking changes and mitigation strategies]

## Implementation Notes

[Important considerations for implementation:]

1. [Technical consideration 1]
2. [Dependency or integration note]
3. [Performance consideration]
4. [Compatibility requirement]

## [Optional: Usage Examples Section]

[If relevant, show how the new functionality will be used]

```python
# Example usage after implementation
new_function = create_something(config)
result = new_function.process(input_data)
```

## Success Criteria

- [ ] [Measurable success criterion 1]
- [ ] [Measurable success criterion 2]
- [ ] [Quality/performance criterion]
- [ ] [Integration/compatibility criterion]
- [ ] [Documentation/testing criterion]

---

## Template Usage Instructions

1. **Title**: Use a descriptive, action-oriented title (e.g., "Add X Feature", "Fix Y Issue", "Refactor Z Component")

2. **Priority Levels**:
   - **High**: Critical issues, performance problems, or frequently used functionality
   - **Medium**: Important improvements that enhance the package but aren't critical
   - **Low**: Nice-to-have features, polish, or long-term improvements

3. **Status Values**:
   - **Pending**: Not yet started
   - **In Progress**: Currently being worked on
   - **Completed**: Finished and tested
   - **On Hold**: Paused for dependencies or other reasons

4. **Commit SHA**: Always include the git commit hash this TODO is based on for tracking

5. **Task Lists**: Use checkboxes for actionable items that can be tracked

6. **Code Examples**: Include current and proposed code where helpful

7. **Design Patterns**: Reference applicable design patterns to guide implementation

8. **Breaking Changes**: Always assess and document potential compatibility impacts

9. **Optional Sections**: Include additional sections like "Enhanced Structure" or "Usage Examples" when they add value

10. **Success Criteria**: Define measurable outcomes to know when the task is complete
