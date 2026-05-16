# VoiceOps AI - Project Structure

## 📁 Directory Layout

```
mansi_mphasis_hackthon_voice_assistance/
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation config
├── MANIFEST.in                    # Package distribution manifest
├── .gitignore                     # Git ignore rules
├── PROJECT_STRUCTURE.md           # This file
├── main.py                        # Application entry point
│
├── voiceops_ai/                   # Main package directory
│   ├── __init__.py               # Package initialization & exports
│   ├── config.py                 # Configuration constants
│   ├── exceptions.py             # Custom exception classes
│   ├── models.py                 # Data models (ConversationTurn, ExtractedEntities)
│   │
│   ├── core/                     # Core processing components
│   │   ├── __init__.py
│   │   ├── regex_processor.py   # Pattern matching & NLU
│   │   ├── voice_input.py       # Speech-to-text handling
│   │   ├── ai_engine.py         # LLM integration (HF + Gemini)
│   │   ├── voice_output.py      # Text-to-speech synthesis
│   │   └── storage.py           # CSV persistence & analytics
│   │
│   ├── agent/                    # Agent orchestration
│   │   ├── __init__.py
│   │   └── orchestrator.py      # Main agentic coordinator
│   │
│   └── utils/                    # Utility functions
│       └── __init__.py
│
├── notebooks/                     # Databricks notebooks
│   └── (exported notebook files)
│
├── tests/                        # Unit tests
│   └── __init__.py
│
└── data/                         # Runtime data (gitignored)
    ├── audio_input/              # Input audio files
    ├── audio_output/             # Generated speech files
    └── logs/                     # Conversation logs & CSV files
```

## 🏗️ Architecture Components

### Core Package (`voiceops_ai/`)

**Configuration Layer**
* `config.py` - Centralized settings, paths, and constants
* `exceptions.py` - Custom exception hierarchy
* `models.py` - Data classes for structured data flow

**Processing Layer (`voiceops_ai/core/`)**
* `regex_processor.py` - 15+ regex patterns for NLU
* `voice_input.py` - Whisper STT integration
* `ai_engine.py` - Dual LLM backend (HuggingFace/Gemini)
* `voice_output.py` - gTTS text-to-speech
* `storage.py` - CSV-based conversation persistence

**Agent Layer (`voiceops_ai/agent/`)**
* `orchestrator.py` - Main coordination engine with agentic routing

## 🔄 Import Structure

### From Root (main.py)
```python
from voiceops_ai import AgentOrchestrator, DEMO_MODE, DEMO_INPUTS
```

### Within Package
```python
# In core modules
from voiceops_ai.config import BASE_DIR
from voiceops_ai.models import ConversationTurn
from voiceops_ai.exceptions import TranscriptionError

# In agent module
from voiceops_ai.core.regex_processor import RegexProcessor
from voiceops_ai.core.storage import ConversationStore
```

## 📦 Installation

### Development Mode
```bash
# From project root
pip install -e .

# Or install directly
python setup.py develop
```

### Production Mode
```bash
pip install .
```

## 🚀 Usage

### As Package
```python
from voiceops_ai import AgentOrchestrator

agent = AgentOrchestrator()
agent.run_demo()
```

### As CLI
```bash
# After installation
voiceops-ai

# Or directly
python main.py
```

## 🧪 Testing

```bash
# Run tests (future implementation)
pytest tests/

# Run with coverage
pytest --cov=voiceops_ai tests/
```

## 📝 Development Guidelines

### Adding New Core Components
1. Create module in `voiceops_ai/core/`
2. Add to `voiceops_ai/core/__init__.py`
3. Export from `voiceops_ai/__init__.py` if public API

### Adding New Agents
1. Create module in `voiceops_ai/agent/`
2. Import core components using relative imports
3. Register in orchestrator if needed

### Configuration Changes
1. Update `voiceops_ai/config.py`
2. Export new constants from `voiceops_ai/__init__.py`
3. Update documentation

## 🎯 Design Principles

1. **Separation of Concerns** - Each module has single responsibility
2. **Package Hierarchy** - Clear distinction between core, agent, utils
3. **Import Organization** - Clean import paths using package structure
4. **Configuration Centralization** - All settings in config.py
5. **Data Isolation** - Runtime data separate from code
6. **Test Coverage** - Separate tests directory
7. **Documentation** - Self-documenting code + comprehensive docs

## 📊 File Purposes

 File/Directory | Purpose |
---------------|---------|
 `main.py` | CLI entry point, argument parsing, main loop |
 `setup.py` | Package metadata and installation config |
 `voiceops_ai/__init__.py` | Public API definition and exports |
 `voiceops_ai/config.py` | All configuration constants |
 `voiceops_ai/exceptions.py` | Error handling hierarchy |
 `voiceops_ai/models.py` | Data transfer objects (DTOs) |
 `voiceops_ai/core/*` | Processing components (no business logic) |
 `voiceops_ai/agent/*` | Business logic and orchestration |
 `data/*` | Runtime artifacts (logs, audio, CSVs) |
 `tests/*` | Unit and integration tests |
 `notebooks/*` | Databricks notebook exports |

---

**Last Updated:** 2026-05-16  
**Version:** 1.0.0
