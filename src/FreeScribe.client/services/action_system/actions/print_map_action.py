import logging
from typing import Any, Dict, Optional
from pathlib import Path
from ..base_action import BaseAction, ActionResult

logger = logging.getLogger(__name__)

class PrintMapAction(BaseAction):
    """
    Action for printing or displaying anatomical maps and medical diagrams.
    
    This action handles intents related to showing visual aids during
    medical consultations.
    """
    
    def __init__(self, maps_directory: Path = None):
        """
        Initialize the print map action.
        
        :param maps_directory: Directory containing anatomical maps and diagrams
        """
        self.maps_directory = maps_directory or Path("assets/medical_maps")
    
    @property
    def action_id(self) -> str:
        return "print_map"
    
    @property
    def display_name(self) -> str:
        return "Show Medical Diagram"
    
    @property
    def description(self) -> str:
        return "Display or print anatomical maps and medical diagrams"
    
    async def can_handle_intent(self, intent_name: str, metadata: Dict[str, Any]) -> bool:
        """
        Check if this action can handle the intent.
        
        :param intent_name: Name of the recognized intent
        :param metadata: Additional data from the intent
        :return: True if this action can handle the intent
        """
        map_related_intents = {
            "show_diagram",
            "print_map",
            "display_anatomy",
            "show_illustration",
            "view_body_part"
        }
        
        # Check if intent directly matches
        if intent_name.lower() in map_related_intents:
            return True
            
        # Check description and required action for map-related keywords
        description = metadata.get("description", "").lower()
        required_action = metadata.get("required_action", "").lower()
        
        map_keywords = {
            "diagram", "map", "illustration", "anatomy",
            "visual", "picture", "drawing", "chart"
        }
        
        return (
            any(keyword in description for keyword in map_keywords) or
            any(keyword in required_action for keyword in map_keywords)
        )
    
    async def execute(self, intent_name: str, metadata: Dict[str, Any]) -> ActionResult:
        """
        Execute the map printing/display action.
        
        :param intent_name: Name of the recognized intent
        :param metadata: Additional data from the intent
        :return: Result of the action execution
        """
        try:
            # Extract relevant parameters
            body_part = metadata.get("parameters", {}).get("body_part", "").lower()
            condition = metadata.get("parameters", {}).get("condition", "").lower()
            
            # In a real implementation, you would:
            # 1. Find the most relevant map based on body_part and condition
            # 2. Load the map file
            # 3. Display it in the UI or send to printer
            
            map_path = self._find_relevant_map(body_part, condition)
            if not map_path:
                return ActionResult(
                    success=False,
                    message="No relevant diagram found for the specified criteria",
                    data={
                        "body_part": body_part,
                        "condition": condition
                    }
                )
            
            return ActionResult(
                success=True,
                message=f"Displaying diagram for {body_part}",
                data={
                    "map_path": str(map_path),
                    "body_part": body_part,
                    "condition": condition,
                    "title": f"Medical diagram: {body_part.title()}",
                    "display_mode": "popup"  # or "print" based on intent
                }
            )
            
        except Exception as e:
            logger.error(f"Error displaying map: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to display map: {str(e)}"
            )
    
    def _find_relevant_map(self, body_part: str, condition: str) -> Optional[Path]:
        """
        Find the most relevant map file based on criteria.
        
        :param body_part: Target body part
        :param condition: Medical condition
        :return: Path to the map file or None if not found
        """
        # This is a simplified implementation
        # In a real system, you would:
        # 1. Have a proper mapping of conditions to diagrams
        # 2. Use fuzzy matching for body parts
        # 3. Consider multiple languages
        # 4. Maybe use embeddings for semantic matching
        
        if not self.maps_directory.exists():
            logger.warning(f"Maps directory not found: {self.maps_directory}")
            return None
            
        # Try different combinations of search patterns
        patterns = [
            f"{body_part}_{condition}.*",
            f"{body_part}.*",
            f"{condition}.*"
        ]
        
        for pattern in patterns:
            matches = list(self.maps_directory.glob(pattern))
            if matches:
                return matches[0]
        
        return None
    
    async def get_ui_data(self) -> Dict[str, Any]:
        """
        Get UI-related data for this action.
        
        :return: Dictionary containing UI configuration
        """
        return {
            "icon": "map",  # FontAwesome icon name
            "color": "#2196F3",  # Blue color for informational actions
            "shortcut": "Ctrl+M",  # Keyboard shortcut
            "priority": 2,  # Medium priority
            "category": "Visual Aids",
            "supported_formats": ["png", "jpg", "pdf", "svg"]
        } 