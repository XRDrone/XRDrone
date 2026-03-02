# Contributing Guide

How to set up, code, test, review, and release so contributions meet our Definition of Done.

## Code of Conduct

Project members will be expected to exhibit behavior as outlined in OSU's Code of Student Conduct  
If any member fails to uphold this standard, please report the act to Student Community Standards  
https://scs.oregonstate.edu/

## Getting Started

Prerequisites: 

- Unity 6000.60f1 LTS 
- Python 3.13.3   
- python -m pip install -r requirements.txt # in inference/pipeline/requirements.txt
 
## Branching & Workflow

Our GitHub workflow is trunk-based. The default branch is main.  
Project member are expected to create a seperate branch and merge it with main at the end of each working day.  
### Naming Conventions
- feature/feature-name  
- user/description  
- bug/description  
### Merge vs Rebase
Members should only rebase their branch if an unrelated change has been made to main that does not conflict with their code.  
Otherwise project members should merge.  

## Project

Maintain project in GitHub for easy tracking of issues and update for additional tasks/assignments.

## Issues & Planning

Explain how to file issues, required templates/labels, estimation, and triage/assignment practices.

All work is tracked through **GitHub Issues** and organized into the **XRDrone Project Board** (`/projects/1`).

Each issue **must use a template** and include description, scope, and acceptance criteria referencing requirements (e.g. `[REQ-003]`).

---

#### Filing Issues

1) **Issues → New Issue → pick template**  
   (Bug Report / Feature Request / Task)

2) Include:

| Field | Expectation |
|---|---|
| **Title** | short + action based (`“Reduce latency below 300 ms [REQ-003]”`) |
| **Description** | concise problem summary + expected vs actual |
| **Steps to Reproduce** | for bugs |
| **Acceptance Criteria** | for features |
| **Screenshots / Logs** | if relevant |

---

#### Labels

| Category | Examples |
|---|---|
| **type** | `type:bug`, `type:feature`, `type:task` |
| **priority** | `prio:P0` (critical) → `P2` (minor) |
| **area** | `area:stream`, `area:vision`, `area:hud`, `area:infra` |
| **status** | `status:in-progress`, `status:blocked`, `status:review` |

---

#### Estimation

Effort levels:

| Code | Meaning |
|---|---|
| `S` | ≤ 1 day |
| `M` | 2–3 days |
| `L` | > 3 days |

Estimation is added by **assignee** and confirmed during **team review**.

---

#### Triage & Assignment

- Weekly triage in **Friday team sync**
- Blocked issues → escalate to mentor if 48h unresolved
- Closed issues **must** link PR (`Closes #42`) and reference ≥1 requirement ID

## Commit Messages

State the convention (e.g., Conventional Commits), include examples, and how to reference issues.

We use the **Conventional Commits** format to maintain clear traceability between commits, pull requests, and project requirements (REQ-001 – REQ-010).

**Format:**  
`<type>(<scope>): <short summary> [REQ-###]`

**Common Types**
- feat – new feature (e.g., streaming, detection, HUD)
- fix – bug fix or patch
- perf – performance or latency optimization
- refactor – non-functional code rework
- test – add or update tests
- docs – documentation or diagrams
- chore – maintenance or build updates

**Examples**
- feat(stream): implement 720p video pipeline [REQ-001]
- perf(stream): maintain ≥24 FPS target for Quest [REQ-002]
- fix(latency): reduce glass-to-glass delay to ≤300 ms [REQ-003]
- feat(vision): enable on-device detection for fire/smoke/humans [REQ-004]
- perf(model): retrain YOLOv8 to reach ≥0.5 F1 on 150 frames [REQ-005]
- docs(ethics): confirm no PII storage, opt-in recording only [REQ-006]
- feat(hud): add altitude and position overlays in VR [REQ-007]
- feat(hud): display FPS, latency, and battery metrics [REQ-008]
- feat(vision): visually mark low-confidence detections [REQ-009]
- fix(stability): auto-recover after forced disconnect [REQ-010]

**Referencing Issues**
Include issue or PR links in the footer:
- Closes #42
- Relates to #57

## Code Style, Linting & Formatting

Name the formatter/linter, config file locations, and the exact commands to check/fix locally.

Due to current implementation limitations, this requirement cannot yet be fully addressed. We’ve documented the constraint and plan to revisit it as the system evolves. 

## Testing

Define required test types, how to run tests, expected coverage thresholds, and when new/updated tests are mandatory.

Due to current implementation limitations, this requirement cannot yet be fully addressed. We’ve documented the constraint and plan to revisit it as the system evolves. 

## Pull Requests & Reviews

<details>
<summary>PR Template</summary>
PR Name

 Description  
 - Briefly describe the purpose of this pull request.
  
 Related Issue  
 - Link to any related GitHub issue(s).
  
 Testing  
 - Describe how you tested your changes (e.g., screenshots, logs, build output).
 
 Checklist
 - [ ] Code builds and runs locally
 - [ ] Feature branch follows naming convention (feature/<description>)
 - [ ] Changes reviewed by at least one teammate
 - [ ] Documentation updated (if needed)
 - [ ] No sensitive data or credentials committed
</details>


Reviewer Expectations:  
Reviewers are expected to run the branch's code on their PC to make sure it is working correctly.  
Reviewers are expected to provide a comment with a reason if they do not approve a PR.  
Reviewers are expected to ensure the branch does not unintentionally remove items from the main branch.  

Approval Rules:  
Pull Request reviewed by one member (besides the one who submitted it)

Required Status Checks:  
Due to current implementation limitations, this requirement cannot yet be fully addressed. We’ve documented the constraint and plan to revisit it as the system evolves.

## CI/CD

Link to pipeline definitions, list mandatory jobs, how to view logs/re-run jobs, and what must pass before merge/release.

Due to current implementation limitations, this requirement cannot yet be fully addressed. We’ve documented the constraint and plan to revisit it as the system evolves.

## Security & Secrets

Do NOT commit: 

- .env files 

- API keys (DJI, YOLO, etc.) 

Secrets go to GitHub Secrets. 

To report vulnerability: DM on Teams. 

## Documentation Expectations

Specify what must be updated (README, docs/, API refs, CHANGELOG) and docstring/comment standards.

All contributions must include relevant documentation updates to keep the project reproducible and maintainable. Documentation changes are part of the Definition of Done and required before merge.

**What to Update**
- README.md — add or revise setup steps, usage instructions, or build/run commands affected by your change
- /docs/ directory — update design diagrams, architecture notes, or benchmark reports (latency, FPS, F1, etc.)
- CHANGELOG.md — add an entry under the current release version describing user-facing changes
- API references — update function/class documentation if new public interfaces are added or modified

**Style Guidelines**
- keep language concise and instructional  
- document why and how — not just what  
- ensure code snippets are executable or valid pseudocode  
- update diagrams/logs in /docs/ whenever architecture, model, or performance metrics change

**Verification**
- every PR must include a doc update if functionality, configuration, or metrics change  
- reviewers verify that doc updates match the implemented behavior and that CHANGELOG entries follow version format:  
  `## [X.Y.Z] – YYYY-MM-DD`

**Definition of Done**
- Builds without errors
- Meets requirement IDs
- Tests pass
- Documentation updated
- Approved PR

## Release Process

Describe versioning scheme, tagging, changelog generation, packaging/publishing steps, and rollback process.

Due to current implementation limitations, this requirement cannot yet be fully addressed. We’ve documented the constraint and plan to revisit it as the system evolves. 

## Support & Contact

Questions? Contact: 
- Teams group chat 
- GitHub Issues (tag as “question”) 
- Typical response time: under 24 hours 
