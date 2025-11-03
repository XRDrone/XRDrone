# Contributing Guide

How to set up, code, test, review, and release so contributions meet our Definition of Done.

## Code of Conduct

Reference the project/community behavior expectations and reporting process.

## Getting Started

Prerequisites: 

- Unity 6000.60f1 LTS (for VR client) 

- Python 3.10 (for YOLO training/inference) 

-YOLOv8 

-Ultralytics  

-OpenCV 
 
Setup: 

git clone https://github.com/.../XRDrone.git   

cd XRDrone   

pip install -r requirements.txt  
pip install --upgrade pip 
pip install ultralytics 
pip install opencv-python 
python3 test_live.py 
python3 -m venv yolovenv 
source yolovenv/bin/activate # for mac 
 .\yolovenv\Scripts\activate # for windows 
 
## Branching & Workflow

Describe the workflow (e.g., trunk-based or GitFlow), default branch, branch naming, and when to rebase vs. merge.

## Issues & Planning

Explain how to file issues, required templates/labels, estimation, and triage/assignment practices.

## Commit Messages

State the convention (e.g., Conventional Commits), include examples, and how to reference issues.

## Code Style, Linting & Formatting

Name the formatter/linter, config file locations, and the exact commands to check/fix locally.

## Testing

Define required test types, how to run tests, expected coverage thresholds, and when new/updated tests are mandatory.

## Pull Requests & Reviews

Outline PR requirements (template, checklist, size limits), reviewer expectations, approval rules, and required status checks.

## CI/CD

Link to pipeline definitions, list mandatory jobs, how to view logs/re-run jobs, and what must pass before merge/release.

## Security & Secrets

Do NOT commit: 

- .env files 

- API keys (DJI, YOLO, etc.) 

Secrets go to GitHub Secrets. 

To report vulnerability: DM on Teams. 

## Documentation Expectations

Specify what must be updated (README, docs/, API refs, CHANGELOG) and docstring/comment standards.

## Release Process

Describe versioning scheme, tagging, changelog generation, packaging/publishing steps, and rollback process.

## Support & Contact

Questions? Contact: 
- Teams group chat 
- GitHub Issues (tag as “question”) 
- Typical response time: under 24 hours 
