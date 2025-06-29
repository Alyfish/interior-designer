# DEVELOPMENT RULES FOR AI ASSISTANT

## 1. Do Not Revert Implemented Features

**Rule:** The AI assistant is strictly forbidden from reverting, altering, or removing features that have been implemented by the user and are working correctly.

**Source of Truth:**
- The primary source of truth is the code located in the `finalobjectdetection/` directory.
- The secondary source of truth is the user's explicit descriptions of implemented features, such as summaries of changes.

**Procedure:**
- Before making any changes, the AI must verify that it is not overwriting a more advanced, user-implemented feature.
- If a file conflict arises, the AI must assume the user's version is the correct one and seek to integrate, not replace.
- When in doubt, ask for clarification instead of reverting to a previous state.

This rule is in place to protect the user's work and ensure forward progress is always maintained. Violation of this rule is a critical error. 