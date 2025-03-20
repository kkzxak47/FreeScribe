"""
Intent action manager for coordinating intent recognition and action execution.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .intents import SpacyIntentRecognizer, Intent
from .actions.print_map import PrintMapAction
from .actions.show_directions import ShowDirectionsAction
from .actions.base import BaseAction

logger = logging.getLogger(__name__)

class IntentActionManager:
    """
    Manages intent recognition and action execution.
    
    This class coordinates between intent recognizers and action handlers,
    providing a unified interface for processing transcribed text.
    """
    
    def __init__(self, maps_directory: Path, google_maps_api_key: str):
        """
        Initialize the intent action manager.
        
        :param maps_directory: Directory to store map images
        :param google_maps_api_key: Google Maps API key for map actions
        """
        self.recognizer = SpacyIntentRecognizer()
        self.actions: Dict[str, BaseAction] = {}
        
        # Initialize recognizer
        self.recognizer.initialize()
        
        # Register actions
        self._register_action(PrintMapAction(maps_directory, google_maps_api_key))
        self._register_action(ShowDirectionsAction())
        
    def _register_action(self, action: BaseAction) -> None:
        """
        Register an action handler.
        
        :param action: Action handler to register
        """
        self.actions[action.action_id] = action
        logger.info(f"Registered action handler: {action.action_id}")
        
    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Process transcribed text to recognize intents and execute actions.
        
        :param text: Transcribed text to process
        :return: List of action results with UI data
        """
        logger.debug(f"Processing text: {text}")
        results = []
        
        # Recognize intents
        intents = self.recognizer.recognize_intent(text)
        logger.debug(f"Intents: {intents}")
        
        # Process each intent
        for intent in intents:
            # Find matching action
            action = self._find_action_for_intent(intent)
            if not action:
                logger.debug(f"No action found for intent: {intent}")
                continue
                
            logger.debug(f"Action found for intent: {action}")
            
            # Execute action
            result = action.execute(intent.name, intent.metadata)
            logger.debug(f"Result: {result}")
            
            if result.success:
                # Add UI data
                ui_data = action.get_ui_data()
                results.append({
                    "action_id": action.action_id,
                    "display_name": action.display_name,
                    "message": result.message,
                    "data": result.data,
                    "ui": ui_data
                })
                
        return results
        
    def _find_action_for_intent(self, intent: Intent) -> Optional[BaseAction]:
        """
        Find an action handler that can handle the given intent.
        
        :param intent: Intent to find handler for
        :return: Matching action handler or None
        """
        for action in self.actions.values():
            logger.debug(f"Checking if {action.action_id} can handle {intent.name}")
            if action.can_handle_intent(intent.name, intent.metadata):
                logger.debug(f"Found matching action: {action.action_id}")
                return action
        logger.debug(f"No action found for intent: {intent.name}")
        return None 