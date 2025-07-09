

# instructions.md

## GitHub Copilot Repo Instructions

This guide standardizes how GitHub Copilot should be used for PRD and task list generation in this repository. Follow these instructions to ensure consistent, actionable outputs.

## General Guidelines

- **Console Syntax:**
Use only PowerShell commands for all console interactions.

Example:

```powershell
# Instead of:
mkdir -p A B C

# Use:
New-Item -ItemType File -Path A, B, C
```

- always use context7
- always use sequentialthinking

## 1. PRD Generation

**Purpose:**
Enable Copilot to create clear, actionable Product Requirements Documents (PRDs) in Markdown, suitable for junior developers.

### When to Trigger

- When a user requests “Generate a PRD” or describes a new feature.


### Process

1. **Receive Initial Prompt:**
User provides a brief feature description or request.
2. **Ask Clarifying Questions:**
Gather details on:
    - Problem/Goal
    - Target user
    - Core functionality
    - User stories
    - Acceptance criteria
    - Scope boundaries (non-goals)
    - Data requirements
    - Design/UI guidelines
    - Potential edge cases
3. **Generate PRD:**
Structure the Markdown file as follows:

4. Introduction / Overview: Feature description, problem, and goal.
5. Goals: List of specific, measurable objectives.
6. User Stories:
Format: “As a [user], I want [action], so that [benefit].”
7. Functional Requirements:
Numbered list of system capabilities.
8. Non-Goals (Out of Scope):
What is *not* included.
9. Design Considerations (Optional):
Mockups, UI guidelines, or style notes.
10. Technical Considerations (Optional):
Dependencies, constraints, integration notes.
11. Success Metrics:
How feature success will be measured.
12. Open Questions:
Remaining clarifications.
1. **Save PRD:**
    - Path: `/tasks/`
    - Filename: `prd-[feature-name].md`

### Final Instructions

- **Do NOT** begin implementation.
- **Always** ask clarifying questions before drafting.
- **Incorporate** user responses before finalizing the PRD.


## 2. Task List Generation

**Purpose:**
Automate creation of a step-by-step task list in Markdown from an existing PRD.

### When to Trigger

- When a user references an existing PRD file.


### Process

1. **Receive PRD Reference:**
Identify which `prd-*.md` file to use.
2. **Analyze PRD:**
Extract user stories, functional requirements, etc.
3. **Phase 1: Generate Parent Tasks:**
    - Create approximately 5 high-level tasks.
    - Present only parent tasks.
    - Pause and prompt:
> “I have generated the high-level tasks based on the PRD. Ready to generate sub-tasks? Respond with ‘Go’ to proceed.”
4. **Upon ‘Go’:**
    - **Phase 2:** Break each parent task into detailed sub-tasks.
    - **Identify Relevant Files:**
List code and test files needed.
5. **Save Task List:**
    - Path: `/tasks/[prd-file-name]/`
    - Filename: `tasks-[prd-file-name].md`

### Output Structure

```markdown
## Relevant Files

- `path/to/file.ts` – Description of purpose
- `path/to/file.test.ts` – Unit tests for this file
...

### Notes

- Testing: `npx jest [optional/path/to/test]`

## Tasks

- [ ] **1.0 Parent Task Title**
  - [ ] 1.1 Sub-task description
  - [ ] 1.2 Sub-task description
- [ ] **2.0 Parent Task Title**
  - [ ] 2.1 Sub-task description
```

