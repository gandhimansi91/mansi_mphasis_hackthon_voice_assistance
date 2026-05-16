#!/usr/bin/env python3
"""
VoiceOps AI - Main Entry Point
Enterprise Voice-Driven Operational Automation System
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from package
from voiceops_ai import (
    AgentOrchestrator,
    DEMO_MODE,
    DEMO_INPUTS,
    VOICE_MODE,
    USE_GEMINI,
    CSV_LOG_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_banner():
    """Display application banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║            🎙️  VOICEOPS AI - VOICE AGENTIC AI ASSISTANT           ║
║                                                                    ║
║        Enterprise Voice-Driven Operational Automation System       ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def display_analytics(agent):
    """Display session analytics"""
    print("\n" + "="*70)
    print("SESSION STATISTICS")
    print("="*70)
    print(f"Session ID: {agent.session_id}")
    print(f"Total turns: {agent.turn_counter}")
    print(f"Notes saved: {len(agent.notes)}")
    print(f"Reminders set: {len(agent.reminders)}")
    print(f"CSV log: {CSV_LOG_PATH}")
    print("="*70 + "\n")


def main():
    """Main application entry point"""
    display_banner()
    
    logger.info("Initializing VoiceOps AI Agent...")
    agent = AgentOrchestrator()
    
    try:
        if DEMO_MODE:
            logger.info("Running in DEMO mode")
            agent.run_demo(DEMO_INPUTS)
        else:
            logger.info("Running in INTERACTIVE mode")
            print("💡 Type your message (or 'quit' to exit)\n")
            
            while agent.running:
                try:
                    user_input = agent.voice_input.get_text_input("\n👤 You: ")
                    
                    if user_input.lower() in ['quit', 'exit', 'stop']:
                        print("\n👋 Goodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    turn = agent.run_turn(user_input)
                    print(f"\n🤖 AI: {turn.ai_response}")
                    
                except KeyboardInterrupt:
                    print("\n\n🛑 Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    print(f"\n❌ Error: {str(e)}")
        
        # Display final analytics
        display_analytics(agent)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
