import logging
from typing import Any, Dict
from ..base_action import BaseAction, ActionResult

logger = logging.getLogger(__name__)

class ScheduleTestAction(BaseAction):
    """
    Action for scheduling medical tests.
    
    This action handles intents related to ordering or scheduling
    medical tests for patients.
    """
    
    @property
    def action_id(self) -> str:
        return "schedule_test"
    
    @property
    def display_name(self) -> str:
        return "Schedule Medical Test"
    
    @property
    def description(self) -> str:
        return "Schedule medical tests and procedures for the patient"
    
    async def can_handle_intent(self, intent_name: str, metadata: Dict[str, Any]) -> bool:
        """
        Check if this action can handle the intent.
        
        :param intent_name: Name of the recognized intent
        :param metadata: Additional data from the intent
        :return: True if this action can handle the intent
        """
        test_related_intents = {
            "request_test",
            "order_lab_work",
            "schedule_procedure",
            "diagnostic_test"
        }
        return (
            intent_name.lower() in test_related_intents or
            (metadata.get("required_action", "").lower().startswith("schedule") and
             "test" in metadata.get("description", "").lower())
        )
    
    async def execute(self, intent_name: str, metadata: Dict[str, Any]) -> ActionResult:
        """
        Execute the test scheduling action.
        
        :param intent_name: Name of the recognized intent
        :param metadata: Additional data from the intent
        :return: Result of the action execution
        """
        try:
            # In a real implementation, this would integrate with your
            # medical scheduling system API
            test_type = metadata.get("parameters", {}).get("test_type", "general")
            urgency = metadata.get("urgency_level", 3)
            
            # Simulate scheduling logic
            logger.info(f"Scheduling {test_type} test with urgency level {urgency}")
            
            return ActionResult(
                success=True,
                message=f"Test scheduled: {test_type}",
                data={
                    "test_type": test_type,
                    "urgency": urgency,
                    "status": "pending",
                    "next_steps": [
                        "Patient will receive notification",
                        "Results will be available in 2-3 days"
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error scheduling test: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to schedule test: {str(e)}"
            )
    
    async def get_ui_data(self) -> Dict[str, Any]:
        """
        Get UI-related data for this action.
        
        :return: Dictionary containing UI configuration
        """
        return {
            "icon": "calendar-plus",  # FontAwesome icon name
            "color": "#4CAF50",  # Green color for medical actions
            "shortcut": "Ctrl+T",  # Keyboard shortcut
            "priority": 1,  # High priority action
            "category": "Medical Procedures"
        } 