# CDSS Source Registry

This document tracks all ingested medical knowledge base sources, their metadata, authority tiers, and ingestion status.

## Ingested Sources

| Source ID | Title | Publisher | Pub Year | Authority Tier | Specialty Tags | Collections | Status |
|---|---|---|---|---|---|---|---|
| `SRC_RHEUM_001` | EULAR Guidelines on Giant Cell Arteritis & Polymyalgia Rheumatica | BMJ / EULAR | 2023 | `guideline` | rheumatology, ophthalmology | `clinical_reference` | Active |
| `SRC_NEURO_001` | Adams and Victor's Principles of Neurology | McGraw-Hill | 2023 | `specialty_textbook` | neurology, ophthalmology | `clinical_reference` | Active |
| `SRC_CARDIO_001` | Harrison's Principles of Internal Medicine - Cardiology | McGraw-Hill | 2022 | `specialty_textbook` | cardiology | `clinical_reference` | Active |

## Metadata Requirements per Source Entry
Each source tagged at intake MUST record:
- `source_id`: Unique alphanumeric tag
- `title`: Full title
- `publisher`: Publishing entity
- `publication_year`: 4-digit year
- `authority_tier`: `guideline` | `specialty_textbook` | `general_textbook` | `patient_education`
- `specialty_tags`: Filterable payload tags
- `license`: Usage and publication rights
