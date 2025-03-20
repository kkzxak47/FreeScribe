import logging
import importlib
import pkgutil
from typing import Dict, List, Type, Optional
from pathlib import Path

from .base_action import BaseAction, ActionResult

logger = logging.getLogger(__name__)

class ActionManager:
    """
    Manages the registration and execution of action plugins.
    
    This class handles:
    - Dynamic loading of action plugins
    - Matching intents to appropriate actions
    - Executing actions based on recognized intents
    """
    
    def __init__(self):
        """Initialize the action manager."""
        self._actions: Dict[str, BaseAction] = {}
    
    async def load_actions_from_directory(self, directory: Path) -> None:
        """
        Load action plugins from a directory.
        
        :param directory: Path to the directory containing action plugins
        """
        try:
            for module_info in pkgutil.iter_modules([str(directory)]):
                module = importlib.import_module(f"{directory.name}.{module_info.name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseAction) and 
                        attr != BaseAction):
                        action_instance = attr()
                        await self.register_action(action_instance)
        except Exception as e:
            logger.error(f"Error loading actions from directory: {e}")
    
    async def register_action(self, action: BaseAction) -> None:
        """
        Register a new action plugin.
        
        :param action: Action instance to register
        """
        if action.action_id in self._actions:
            logger.warning(f"Action with ID {action.action_id} already registered")
            return
        
        self._actions[action.action_id] = action
        logger.info(f"Registered action: {action.display_name} ({action.action_id})")
    
    async def get_actions_for_intent(self, intent_name: str, metadata: dict) -> List[BaseAction]:
        """
        Find all actions that can handle a given intent.
        
        :param intent_name: Name of the recognized intent
        :param metadata: Additional data from the intent
        :return: List of actions that can handle the intent
        """
        matching_actions = []
        for action in self._actions.values():
            if await action.can_handle_intent(intent_name, metadata):
                matching_actions.append(action)
        return matching_actions
    
    async def execute_action(self, action_id: str, intent_name: str, metadata: dict) -> Optional[ActionResult]:
        """
        Execute a specific action.
        
        :param action_id: ID of the action to execute
        :param intent_name: Name of the recognized intent
        :param metadata: Additional data from the intent
        :return: Result of the action execution
        """
        action = self._actions.get(action_id)
        if not action:
            logger.error(f"Action not found: {action_id}")
            return None
        
        try:
            return await action.execute(intent_name, metadata)
        except Exception as e:
            logger.error(f"Error executing action {action_id}: {e}")
            return ActionResult(
                success=False,
                message=f"Error executing action: {str(e)}"
            )
    
    def get_all_actions(self) -> List[BaseAction]:
        """
        Get all registered actions.
        
        :return: List of all registered actions
        """
        return list(self._actions.values()) 