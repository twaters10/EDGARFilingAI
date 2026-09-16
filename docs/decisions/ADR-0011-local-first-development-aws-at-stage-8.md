# ADR-0011: Local-first development, AWS at Stage 8

**Status:** Accepted  
**Stage:** 1-8

## Context

The architecture targets Athena and OpenSearch Serverless. Building directly against them during the stages where the most learning happens means slow iteration, cloud spend while experimenting, and CI that needs credentials.

## Decision

Stages 1-7 run on DuckDB and Docker OpenSearch. AWS is an explicit porting stage (Stage 8) behind a deliberately narrow backend interface, with a conformance suite that runs against both backends.

## Alternatives considered

AWS-first costs iteration speed exactly when it is most needed.

## Consequences

Fast iteration, free CI, no spend while learning. The port is real work (~2 days) — Trino vs DuckDB dialect, async submit/poll vs in-process, Glue catalog, SigV4 — but it is contained by the interface.
