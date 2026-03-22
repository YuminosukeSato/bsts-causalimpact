# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in bsts-causalimpact, please report it responsibly.

### How to Report

Email the maintainer directly rather than opening a public issue:

- Email: Open a private security advisory on GitHub (Settings > Security > Advisories > New)

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- Acknowledgment within 48 hours
- Status update within 7 days
- Fix released as soon as practical

## Scope

This library processes numerical data (time series) and does not handle authentication, network requests, or sensitive user data directly. Security concerns are most likely to involve:

- Denial of service via crafted input data
- Unsafe memory access in the Rust extension
- Dependency vulnerabilities
