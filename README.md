# Paper Digest - Security Research Tool

This is a comprehensive tool for discovering the latest security research papers from ArXiv.

## Overview

Every day at 6 AM (SGT), this tool automatically fetches new security-related papers from ArXiv, analyzes them using a two-stage AI assessment process, and publishes relevant papers to RSS feeds:

- **AI Security Feed**: [https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/rss.xml](https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/rss.xml)
- **Web3 Security Feed**: [https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/web3_security_rss.xml](https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/web3_security_rss.xml)

### AI Security Feed
Focuses on papers related to AI security, safety, and red teaming, including topics like LLM jailbreaking, prompt injection, adversarial attacks, model extraction, data poisoning, privacy attacks, alignment, robustness, and safety evaluation.

### Web3 Security Feed
Focuses on papers related to Web3, blockchain, and smart contract security, including topics like smart contract vulnerabilities, DeFi security, blockchain consensus security, cryptographic protocols, zero-knowledge proofs, Web3 privacy, cryptocurrency security, smart contract auditing, and formal verification.

## Features

- **Paper Discovery**: Fetch and process papers from ArXiv
- **Two-Stage Assessment**: Cost-efficient filtering with a quick assessment followed by detailed analysis
- **Intelligent Filtering**: AI-powered relevance assessment and tagging
- **RSS Feed Generation**: Generate RSS feeds of relevant papers
- **Metadata Storage**: Store papers in Zotero library via API
- **Automated Workflow**: Runs daily via GitLab CI/CD

## Setup

The RSS feeds are publicly available (see links above), but you may want to try with other research topics or system prompts. The following is the step-by-step to replicate the service.

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure your `.env` file has the following variables:

```
API_KEY=your_llm_api_key
BASE_URL=https://your-litellm-host/v1  # or https://openrouter.ai/api/v1
ZOTERO_LIBRARY_ID=your_zotero_library_id
ZOTERO_API_KEY=your_zotero_api_key
ZOTERO_LIBRARY_TYPE=user
AI_SECURITY_RSS_URL=your_ai_security_rss_feed_url
WEB3_SECURITY_RSS_URL=your_web3_security_rss_feed_url
QUICK_ASSESSMENT_MODEL=openai/gpt-4.1-nano  # Model for initial quick filtering
DETAILED_ASSESSMENT_MODEL=openai/gpt-4.1-mini  # Model for detailed analysis
TEMPERATURE=0.1              # Optional: specify the temperature (0.0 to 1.0)
```

## Usage

### Update RSS Feeds from ArXiv

Generate AI security feed:
```bash
python update_rss.py --feed-type ai-security
```

Generate Web3 security feed:
```bash
python update_rss.py --feed-type web3-security
```

Generate both feeds (default is AI security):
```bash
python update_rss.py  # Generates AI security feed
python update_rss.py --feed-type web3-security  # Generates Web3 security feed
```

This will:

1. Fetch recent papers from ArXiv feeds (cs.AI, cs.LG, cs.CL, cs.CV) and ACL Anthology
2. Perform a quick initial assessment using a cost-efficient model
3. Perform detailed assessment on promising papers using a more powerful model
4. Store relevant papers in Zotero collections (`AI Security Papers` or `Web3 Security Papers`)
5. Generate the appropriate RSS feed (`rss.xml` or `web3_security_rss.xml`)

**Cost Optimization**: The two-stage assessment process significantly reduces costs:
- **Initial Filtering**: Uses a free or very low-cost model (e.g., gpt-4.1-nano)
- **Detailed Analysis**: Only papers that pass initial filtering are processed with a more expensive model
- **Cost Savings**: Detailed cost breakdown is provided in the output

## Integration with Existing System

This tool integrates with:

- **LLM API**: For AI processing and relevance assessment (OpenRouter, LiteLLM, or any OpenAI-compatible API)
- **ArXiv**: For preprint discovery
- **Zotero**: For paper metadata storage and organization
- **GitLab CI/CD**: For automated daily execution

## Files Overview

- `update_rss.py`: Update RSS feed from ArXiv with two-stage assessment
- `utils/llm.py`: AI processing and rate limiting
- `utils/zotero.py`: Zotero API operations
- `.gitlab-ci.yml`: GitLab CI/CD pipeline for daily execution
