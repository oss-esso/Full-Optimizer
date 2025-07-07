# GitHub Copilot Instruction Guide

## General

- **Console Syntax**  
  Always use PowerShell‐only commands when interacting with the console.  
  - **Example:**  
    ```powershell
    # Instead of:
    mkdir -p A B C

    # Use:
    New-Item -ItemType File -Path A, B, C
    ```

---

## 1. PRD Generation

**Goal:**  
Guide Copilot to produce a clear, actionable Product Requirements Document (PRD) in Markdown, suitable for a junior developer.

### Trigger
When the user requests “Generate a PRD” or provides a new feature prompt.

### Process
1. **Receive Initial Prompt**  
   User gives a brief description or request for a new feature.
2. **Ask Clarifying Questions**  
   Collect details about:
   - Problem/Goal
   - Target user
   - Core functionality
   - User stories
   - Acceptance criteria
   - Scope boundaries (non‑goals)
   - Data requirements
   - Design/UI guidelines
   - Potential edge cases
3. **Generate PRD**  
   Structure the Markdown document as follows:
   1. **Introduction / Overview**  
      Brief description of the feature and problem it solves. State the goal.
   2. **Goals**  
      List specific, measurable objectives.
   3. **User Stories**  
      “As a [user], I want [action], so that [benefit].”
   4. **Functional Requirements**  
      Numbered list of clear, concise system capabilities.
   5. **Non‑Goals (Out of Scope)**  
      What this feature will *not* include.
   6. **Design Considerations** _(Optional)_  
      Links or notes on mockups, UI guidelines, style.
   7. **Technical Considerations** _(Optional)_  
      Dependencies, constraints, integration notes.
   8. **Success Metrics**  
      How to measure feature success.
   9. **Open Questions**  
      Remaining clarifications needed.
4. **Save PRD**  
   - Path: `/tasks/`  
   - Filename: `prd-[feature-name].md`

### Final Instructions
- **Do NOT** start implementation.
- **Always** ask clarifying questions first.
- **Incorporate** user responses before finalizing the PRD.

---

## 2. Task List Generation

**Goal:**  
Automate creation of a stepped task list in Markdown from an existing PRD.

### Trigger
When the user references an existing PRD file.

### Process
1. **Receive PRD Reference**  
   Identify which `prd-*.md` file to read.
2. **Analyze PRD**  
   Extract user stories, functional requirements, etc.
3. **Phase 1: Generate Parent Tasks**  
   - Create ~5 high‑level tasks.
   - Present only parent tasks.
   - Pause and prompt:  
     > “I have generated the high‑level tasks based on the PRD. Ready to generate sub‑tasks? Respond with ‘Go’ to proceed.”
4. **Upon “Go”**  
   - **Phase 2:** Break each parent task into detailed sub‑tasks.
   - **Identify Relevant Files**  
     List code and test files needed.
5. **Save Task List**  
   - Path: `/tasks/[prd-file-name]/`  
   - Filename: `tasks-[prd-file-name].md`

### Output Structure

```markdown
## Relevant Files

- `path/to/file.ts` – Description of purpose  
- `path/to/file.test.ts` – Unit tests for this file  
…

### Notes

- Testing: `npx jest [optional/path/to/test]`

## Tasks

- [ ] **1.0 Parent Task Title**
  - [ ] 1.1 Sub‑task description
  - [ ] 1.2 Sub‑task description
- [ ] **2.0 Parent Task Title**
  - [ ] 2.1 Sub‑task description
…
