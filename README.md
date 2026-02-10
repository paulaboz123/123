1) Customer acceptance and scope

Acceptance criteria met: every acceptance criterion is satisfied, or any deviation is explicitly documented and accepted.

Customer sign-off recorded: confirmation is captured in the ticket (comment/approval) or documented outcome of a demo.

No hidden work: any follow-ups, improvements, or tech debt are logged as separate tickets (with clear ownership).

2) Code/Change quality (engineering baseline)

Peer review completed:

At least one reviewer not involved in the implementation.

Review covers correctness, maintainability, edge cases, and operational impact.

Style and correctness gates pass (as applicable):

Lint/format checks pass.

Static analysis (if present) passes.

No secrets committed; config follows team standards.

No critical known defects:

Known limitations are documented.

No open “must-fix” issues remain on the ticket.

3) Verification (proof it works)

At least one of the following is true, and evidence is attached:

Automated tests (preferred):

Relevant unit/integration tests added or updated.

CI pipeline is green.

Manual verification (acceptable for small teams if documented):

A short step-by-step checklist exists in the ticket.

Results captured (what was tested, outcome, environment/version).

For risky changes:

Smoke test run on the target environment (or staging equivalent).

4) Delivered to the agreed environment(s)

Deployed or delivered to the environment agreed for “Done” (e.g., staging or production).

Ticket includes:

Version/build identifier (tag/commit/hash).

Where to access it (URL/endpoint/dashboard/location).

If production is not part of scope:

A clear note states: “Done = ready for prod” vs “Done = shipped to prod”.

5) Operational readiness (minimum viable support)

How to observe it:

Link to logs/metrics/dashboard or explicit note that monitoring is not available yet and what the risk is.

How to roll back / disable:

Rollback steps, or feature flag strategy, or safe revert path.

Failure modes considered:

Basic error handling is in place.

If it fails, it fails safely (no silent corruption, no endless retries without visibility).

6) Documentation (lightweight but real)

A concise note exists (ticket or repo), covering:

What changed and why (1–3 bullets).

How to use it (inputs/outputs, examples, links).

Known limitations and next steps.

Ownership/contacts for follow-up (if needed).

7) Security and compliance (when applicable)

Data handling validated:

PII/credentials/secrets handled per policy.

Least-privilege access applied (roles/service accounts).

Security checks:

Dependency/vulnerability scanning results reviewed if your pipeline runs them.

No new critical vulnerabilities introduced (or explicitly accepted with mitigation plan).
