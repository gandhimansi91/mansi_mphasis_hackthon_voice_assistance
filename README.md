# 🎙️ VoiceOps AI - Voice Agentic AI Assistant

**Enterprise Voice-Driven Operational Automation System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Databricks](https://img.shields.io/badge/Databricks-Compatible-orange.svg)](https://databricks.com/)

## 🌟 Overview

VoiceOps AI is an enterprise-grade Voice Agentic AI Assistant designed to automate and simplify operational workflows using voice-driven interactions. Built for hackathon demonstration, this system showcases production-quality Python engineering, advanced regex patterns, GenAI integration, and intelligent agentic behavior.

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     VOICEOPS AI ASSISTANT                          │
├────────────────────────────────────────────────────────────────────┤
│  INPUT LAYER          │  PROCESSING LAYER      │  OUTPUT LAYER     │
├───────────────────────┼────────────────────────┼───────────────────┤
│                       │                        │                   │
│  ┌──────────────┐     │  ┌──────────────┐     │  ┌──────────────┐ │
│  │ Voice Input  │────>│  │ RegEx Engine │────>│  │   AI LLM     │ │
│  │ (Whisper STT)│     │  │  - Intent    │     │  │  (Flan-T5)   │ │
│  └──────────────┘     │  │  - Entities  │     │  └──────────────┘ │
│         OR            │  │  - Dates     │     │         │         │
│  ┌──────────────┐     │  │  - Emails    │     │         ▼         │
│  │ Text Fallback│     │  │  - Phones    │     │  ┌──────────────┐ │
│  └──────────────┘     │  └──────────────┘     │  │ Voice Output │ │
│                       │         │              │  │    (gTTS)    │ │
│                       │         ▼              │  └──────────────┘ │
│                       │  ┌──────────────┐     │         │         │
│                       │  │   Agentic    │     │         ▼         │
│                       │  │ Orchestrator │     │  ┌──────────────┐ │
│                       │  │ - Routing    │     │  │  CSV Logger  │ │
│                       │  │ - Context    │     │  │  Analytics   │ │
│                       │  │ - History    │     │  └──────────────┘ │
│                       │  └──────────────┘     │                   │
└───────────────────────┴────────────────────────┴───────────────────┘
```

## 🎯 Core Capabilities

* **Voice Recognition**: OpenAI Whisper-based speech-to-text (tiny model, CPU-optimized)
* **Intent Detection**: 15+ regex patterns for accurate intent classification
* **Entity Extraction**: Dates, emails, phones, keywords using advanced regex
* **Agentic Routing**: Context-aware task distribution and decision-making
* **LLM Integration**: Google Flan-T5-base (local) + Gemini API (optional)
* **Voice Synthesis**: Google Text-to-Speech (gTTS)
* **Persistent Memory**: CSV-based conversation logging and analytics
* **Real-time Analytics**: Intent distribution, keyword frequency, performance metrics

## 📋 Supported Intents

1. **greeting** - Hello, hi, good morning/afternoon/evening
2. **task_query** - Show tasks, list todos, what do I have to do
3. **note_save** - Save note, remember this, write down
4. **summarize** - Summarize, give overview, brief recap
5. **reminder_set** - Remind me, set reminder, alert me
6. **question** - What is, explain, tell me about
7. **farewell** - Goodbye, bye, exit
8. **unknown** - Fallback for unrecognized intents

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mansi_mphasis_hackthon_voice_assistance

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Run the main application
python main.py
```

Or in Databricks:

```python
# Open the notebook: "Voice Agentic AI Assistant - VoiceOps AI"
# Run all cells sequentially
```

## 📁 Project Structure

```
mansi_mphasis_hackthon_voice_assistance/
├── config.py              # Configuration and constants
├── exceptions.py          # Custom exception classes
├── models.py              # Data models (ConversationTurn, ExtractedEntities)
├── regex_processor.py     # RegEx engine (15+ patterns)
├── voice_input.py         # Voice/text input handler (Whisper STT)
├── ai_engine.py           # AI response engine (Flan-T5 + Gemini)
├── voice_output.py        # Text-to-speech handler (gTTS)
├── storage.py             # CSV storage and analytics
├── orchestrator.py        # Agentic orchestrator (main brain)
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── Voice Agentic AI Assistant - VoiceOps AI  # Databricks notebook
```

## 🔧 Configuration

Edit `config.py` to customize behavior:

```python
# Mode flags
VOICE_MODE = False  # Set to True for audio file input
USE_GEMINI = False  # Set to True to use Gemini API
DEMO_MODE = True    # Runs pre-defined demo scenarios

# API keys
GEMINI_API_KEY = "your-key-here"  # Optional

# Paths
BASE_DIR = Path("/dbfs/tmp/voiceops_ai/")  # Change for local usage

# Model configs
HF_MODEL_NAME = "google/flan-t5-base"
WHISPER_MODEL_NAME = "tiny"
```

## 🧪 Testing Individual Modules

Each module can be tested independently:

```bash
# Test RegEx processor
python regex_processor.py

# Test AI engine
python ai_engine.py

# Test voice input handler
python voice_input.py

# Test storage
python storage.py
```

## 🏆 Hackathon Evaluation Coverage

| Criteria | Implementation | File(s) |
|----------|----------------|----------|
| **Python Skills** | OOP, type hints, dataclasses, decorators, generators | All modules |
| **RegEx Mastery** | 15+ patterns (intents, dates, emails, phones) | `regex_processor.py` |
| **GenAI Integration** | Flan-T5-base (local) + Gemini API stub | `ai_engine.py` |
| **Voice I/O** | Whisper STT + gTTS TTS | `voice_input.py`, `voice_output.py` |
| **Data Handling** | Pandas CSV operations, analytics, Counter | `storage.py`, `main.py` |
| **Agentic Behavior** | Intent routing, context management, history | `orchestrator.py` |
| **Code Quality** | Docstrings, logging, error handling, modularity | All modules |
| **Advanced Python** | `@lru_cache`, `defaultdict`, `filter()`, `map()`, `sorted()` | Multiple files |

## 🔍 Key Features

### 1. Advanced RegEx Patterns (15+)

* **Date patterns (5)**: DD/MM/YYYY, "15th June", "today", "next Monday"
* **Email pattern (1)**: RFC-compliant email extraction
* **Phone patterns (3)**: International (+91), 10-digit, US format
* **Intent patterns (8 categories, 15+ patterns)**: greeting, farewell, task_query, note_save, summarize, reminder_set, question

### 2. Agentic Decision-Making

* Context-aware intent routing
* Time-based greeting responses
* Dynamic task/note/reminder management
* Conversation history tracking (last 5 turns)

### 3. Production-Grade Features

* Custom exception hierarchy
* Comprehensive logging
* Type hints and docstrings
* Dataclasses for structured data
* Functional programming patterns (`filter`, `map`, `sorted`)
* Caching with `@lru_cache`
* Generator functions for demo simulation

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributors

* **Hackathon Team** - Initial development
* **Databricks Community** - Platform support

## 🙏 Acknowledgments

* OpenAI Whisper for speech-to-text
* Google for Flan-T5 and gTTS
* HuggingFace for transformers library
* Databricks for notebook platform

---

**Built with ❤️ for Hackathon 2024**

*VoiceOps AI - Making operational workflows conversational*