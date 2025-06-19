# retrieve_all_papers.py - Paper Retrieval Script

A comprehensive script to retrieve all papers from the Qdrant knowledge base with multiple output formats.

## Overview

This script provides a powerful interface to extract and analyze all papers stored in your Qdrant knowledge base. It supports multiple output formats and can handle large datasets efficiently.

## Usage

### Basic Usage

```bash
# Get paper count only
python retrieve_all_papers.py --count-only

# Display all papers in a formatted view (default)
python retrieve_all_papers.py

# Export to JSON format
python retrieve_all_papers.py --format json --output papers.json

# Export to CSV format
python retrieve_all_papers.py --format csv --output papers.csv

# Display with custom output file
python retrieve_all_papers.py --output papers.txt
```

### Command Line Options

- `--format {display,json,csv}`: Output format (default: display)
- `--output OUTPUT, -o OUTPUT`: Output file path (optional)
- `--count-only`: Only show the count of papers
- `--help, -h`: Show help message

## Output Formats

### 1. Display Format (Default)

Human-readable formatted output showing:

- Paper title, authors, URL, date
- Modalities (text, image, audio, etc.)
- Relevance score
- Abstract (truncated for readability)
- Code repository links
- Paper type and tags

Example:

```
================================================================================
ALL PAPERS IN QDRANT KNOWLEDGE BASE (2816 papers)
================================================================================

--------------------------------------------------------------------------------
Paper #1
--------------------------------------------------------------------------------
Title: Open AI and its Impact on Fraud Detection in Financial Industry
Authors: Sina Ahmadi
URL: https://openalex.org/W4398632649
Date: 2024-01-01T00:00:00
Abstract: As per the Nilson report, fraudulent activities targeting cards amounted to a loss of
 $32.34 billion globally in 2021, 14 % increase from previous year...
```

### 2. JSON Format

Structured JSON output for programmatic use:

```json
[
  {
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "url": "https://example.com",
    "date": "2024-01-01T00:00:00",
    "modalities": ["text", "image"],
    "relevance_score": 5,
    "abstract": "Paper abstract...",
    "code_repository": "",
    "paper_type": "Research Paper",
    "tags": ["AI security", "adversarial attacks"]
  }
]
```

### 3. CSV Format

Comma-separated values for spreadsheet applications:

```csv
title,authors,url,date,modalities,relevance_score,abstract,code_repository,paper_type,tags
"Paper Title","Author 1;Author 2","https://example.com","2024-01-01T00:00:00","text;image",5,"Abstract...","","Research Paper","AI security;adversarial attacks"
```

## Features

- **Efficient Retrieval**: Handles large datasets (tested with 2,816+ papers)
- **Multiple Formats**: Support for display, JSON, and CSV outputs
- **Comprehensive Metadata**: Includes all paper information stored in Qdrant
- **File Export**: Save output to files for further analysis
- **Error Handling**: Robust error handling and informative messages
- **Flexible Usage**: Command-line interface with multiple options

## Requirements

- Python 3.6+
- Required packages (from requirements.txt):
  - qdrant-client
  - python-dotenv

## Environment Setup

Ensure your `.env` file contains:

```
QDRANT_API_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_cloud_api_key
```

## Examples

### Quick Analysis

```bash
# Get total paper count
python retrieve_all_papers.py --count-only
# Output: Total papers in knowledge base: 2816
```

### Data Export for Analysis

```bash
# Export to JSON for programmatic analysis
python retrieve_all_papers.py --format json --output papers_analysis.json

# Export to CSV for spreadsheet analysis
python retrieve_all_papers.py --format csv --output papers_spreadsheet.csv
```

### Documentation

```bash
# Save formatted output to a text file
python retrieve_all_papers.py --output paper_documentation.txt
```

## Integration

This script uses the existing Qdrant setup in `utils/qdrant.py` and leverages the `get_all_papers()` function to retrieve data from your knowledge base.
