"""
Map and directions action implementation using Google Maps API.
"""

import logging
from typing import Any, Dict, Optional, List
from pathlib import Path
import googlemaps
from datetime import datetime
import os
from .base import BaseAction, ActionResult

logger = logging.getLogger(__name__)

class PrintMapAction(BaseAction):
    """Action to display maps and directions using Google Maps."""
    
    def __init__(self, maps_directory: Path, google_maps_api_key: str):
        """
        Initialize the map action with a directory for storing maps.
        
        :param maps_directory: Directory to store generated maps
        :param google_maps_api_key: Google Maps API key for authentication
        """
        self.maps_directory = maps_directory
        self.maps_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Google Maps client
        self.gmaps = googlemaps.Client(key=google_maps_api_key)
        
        # Cache for storing location results
        self._location_cache = {}
        
        # Mock database of locations - in a real app this would come from a proper database
        self.locations = {
            "radiology": {
                "floor": 2,
                "wing": "East",
                "landmarks": ["Main Elevator", "Waiting Area", "Reception"],
                "directions": "Take the main elevator to the 2nd floor, turn right, and follow signs to Radiology"
            },
            "emergency": {
                "floor": 1,
                "wing": "North",
                "landmarks": ["Main Entrance", "Triage", "Ambulance Bay"],
                "directions": "Enter through the main entrance, Emergency is directly ahead"
            },
            "cafeteria": {
                "floor": 1,
                "wing": "West",
                "landmarks": ["Gift Shop", "Main Hallway", "Vending Machines"],
                "directions": "From the main entrance, follow the hallway past the gift shop"
            }
        }

    @property
    def action_id(self) -> str:
        """Get the unique identifier for this action."""
        return "print_map"

    @property
    def display_name(self) -> str:
        """Get the human-readable name for this action."""
        return "Print Map"

    @property
    def description(self) -> str:
        """Get the detailed description of what this action does."""
        return "Display maps and provide directions to hospital locations using Google Maps"

    def can_handle_intent(self, intent_name: str, metadata: Dict[str, Any]) -> bool:
        """Check if this action can handle the given intent."""
        if intent_name not in ["show_map", "get_directions", "find_location"]:
            return False
            
        # Check if we have location information for this intent
        if "destination" in metadata:
            dest = metadata["destination"].lower()
            return any(loc in dest for loc in self.locations.keys())
            
        return False

    def execute(self, intent_name: str, metadata: Dict[str, Any]) -> ActionResult:
        """Execute the action for the given intent."""
        dest = metadata["destination"].lower()
        location = None
        
        # Find matching location
        for loc_name, loc_data in self.locations.items():
            if loc_name in dest:
                location = loc_data
                location_name = loc_name
                break
                
        if not location:
            return ActionResult(
                success=False,
                message="Sorry, I don't have information about that location.",
                data={}
            )
            
        try:
            # Search Google Maps for the location
            search_query = f"{location_name} hospital"
            if search_query not in self._location_cache:
                places_result = self.gmaps.places(search_query)
                if not places_result.get('results'):
                    return ActionResult(
                        success=False,
                        message=f"Could not find {location_name} on Google Maps.",
                        data={}
                    )
                self._location_cache[search_query] = places_result['results'][0]
            
            place = self._location_cache[search_query]
            
            # Generate static map
            map_filename = f"{location_name}_map.png"
            map_path = self.maps_directory / map_filename
            
            # Create static map URL manually
            lat = place['geometry']['location']['lat']
            lng = place['geometry']['location']['lng']
            static_map_url = (
                f"https://maps.googleapis.com/maps/api/staticmap?"
                f"center={lat},{lng}&"
                f"zoom=15&"
                f"size=640x640&"
                f"markers=color:red%7C{lat},{lng}&"
                f"key={self.gmaps.key}"
            )
            
            # Download and save the map
            import requests
            try:
                response = requests.get(static_map_url)
                response.raise_for_status()  # Raise an exception for bad status codes
                with open(map_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully saved map to {map_path}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download map: {str(e)}")
                return ActionResult(
                    success=False,
                    message="Failed to generate map image.",
                    data={"error": str(e)}
                )
            
            # Generate response based on intent type
            if intent_name == "show_map":
                message = f"Here's a map of the {location_name} area"
            elif intent_name == "get_directions":
                # Get directions
                directions_result = self.gmaps.directions(
                    origin="current location",  # In a real app, this would be the user's location
                    destination=place['formatted_address'],
                    mode="driving",
                    departure_time=datetime.now()
                )
                
                if directions_result:
                    route = directions_result[0]
                    duration = route['legs'][0]['duration']['text']
                    distance = route['legs'][0]['distance']['text']
                    steps = [step['html_instructions'] for step in route['legs'][0]['steps']]
                    
                    message = f"Here are directions to {location_name}: {duration} ({distance})"
                else:
                    message = f"Here are directions to {location_name}: {location['directions']}"
            else:  # find_location
                message = f"The {location_name} is located on floor {location['floor']} in the {location['wing']} wing"
            
            return ActionResult(
                success=True,
                message=message,
                data={
                    "title": f"{location_name.title()} Information",
                    "additional_info": {
                        "floor": f"Floor {location['floor']}",
                        "wing": f"{location['wing']} Wing",
                        "key_landmarks": location["landmarks"],
                        "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}",
                        "map_image_path": str(map_path)
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing map action: {str(e)}")
            return ActionResult(
                success=False,
                message="Failed to generate map image.",
                data={"error": str(e)}
            )

    def get_ui_data(self) -> Dict[str, Any]:
        """Get UI configuration for displaying results."""
        return {
            "icon": "🗺️",
            "color": "#4CAF50"
        }

    def _find_relevant_map(self, destination: str, transport_mode: str) -> Optional[Path]:
        """
        Find the most relevant map file based on destination.
        
        :param destination: Target medical facility
        :param transport_mode: Mode of transport (driving, transit, walking)
        :return: Path to the map file or None if not found
        """
        if not self.maps_directory.exists():
            logger.warning(f"Maps directory not found: {self.maps_directory}")
            return None
            
        # Try different combinations of search patterns
        patterns = [
            f"{destination}_{transport_mode}.*",
            f"{destination}.*",
            "default_map.*"
        ]
        
        for pattern in patterns:
            matches = list(self.maps_directory.glob(pattern))
            if matches:
                return matches[0]
        
        return None
    
    def _get_travel_time(self, destination: str, transport_mode: str) -> str:
        """Get estimated travel time to destination."""
        # In a real implementation, this would:
        # 1. Use a mapping service API (Google Maps, etc.)
        # 2. Consider current traffic conditions
        # 3. Account for time of day
        return "25-30 minutes"  # Placeholder
    
    def _get_landmarks(self, destination: str) -> List[str]:
        """Get key landmarks along the route."""
        # In a real implementation, this would:
        # 1. Load from a database of known landmarks
        # 2. Filter by relevance to the route
        return [
            "Turn right at Central Hospital",
            "Pass the Main Street intersection",
            "Look for the blue hospital sign"
        ] 