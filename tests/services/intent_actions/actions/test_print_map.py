"""
Unit tests for the print map action.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any
from services.intent_actions.actions.print_map import PrintMapAction
from services.intent_actions.actions.base import ActionResult

@pytest.fixture
def mock_gmaps():
    """Create a mock Google Maps client."""
    with patch('googlemaps.Client') as mock:
        client = Mock()
        client.key = "fake_api_key"  # Set the API key explicitly
        mock.return_value = client
        yield client

@pytest.fixture
def mock_requests():
    """Create a mock requests module."""
    with patch('requests.get') as mock:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'fake_image_data'
        mock.return_value = mock_response
        yield mock

@pytest.fixture
def maps_dir(tmp_path):
    """Create a temporary directory for maps."""
    maps_dir = tmp_path / "hospital_maps"
    maps_dir.mkdir()
    return maps_dir

@pytest.fixture
def print_map_action(maps_dir, mock_gmaps):
    """Create a PrintMapAction instance with mocked dependencies."""
    return PrintMapAction(maps_dir, "fake_api_key")

def test_action_properties(print_map_action):
    """Test that action properties return correct values."""
    assert print_map_action.action_id == "print_map"
    assert print_map_action.display_name == "Print Map"
    assert "Google Maps" in print_map_action.description

def test_can_handle_intent(print_map_action):
    """Test intent handling capability."""
    # Test valid intents
    assert print_map_action.can_handle_intent("show_map", {"destination": "radiology"}) is True
    assert print_map_action.can_handle_intent("get_directions", {"destination": "emergency"}) is True
    assert print_map_action.can_handle_intent("find_location", {"destination": "cafeteria"}) is True
    
    # Test invalid intents
    assert print_map_action.can_handle_intent("invalid_intent", {"destination": "radiology"}) is False
    assert print_map_action.can_handle_intent("show_map", {}) is False
    assert print_map_action.can_handle_intent("show_map", {"destination": "unknown"}) is False

def test_execute_show_map(print_map_action, mock_gmaps, mock_requests):
    """Test executing show_map intent."""
    # Mock Google Places API response
    mock_gmaps.places.return_value = {
        'results': [{
            'geometry': {'location': {'lat': 37.7749, 'lng': -122.4194}},
            'place_id': 'test_place_id',
            'formatted_address': '123 Test St'
        }]
    }
    
    result = print_map_action.execute("show_map", {"destination": "radiology"})
    
    assert isinstance(result, ActionResult)
    assert result.success is True
    assert "map of the radiology area" in result.message
    assert "map_image_path" in result.data["additional_info"]
    assert "google_maps_url" in result.data["additional_info"]
    
    # Verify map was saved
    map_path = Path(result.data["additional_info"]["map_image_path"])
    assert map_path.exists()
    assert map_path.read_bytes() == b'fake_image_data'

def test_execute_get_directions(print_map_action, mock_gmaps, mock_requests):
    """Test executing get_directions intent."""
    # Mock Google Places API response
    mock_gmaps.places.return_value = {
        'results': [{
            'geometry': {'location': {'lat': 37.7749, 'lng': -122.4194}},
            'place_id': 'test_place_id',
            'formatted_address': '123 Test St'
        }]
    }
    
    # Mock Google Directions API response
    mock_gmaps.directions.return_value = [{
        'legs': [{
            'duration': {'text': '30 mins'},
            'distance': {'text': '5 km'},
            'steps': [{'html_instructions': 'Turn right'}]
        }]
    }]
    
    result = print_map_action.execute("get_directions", {"destination": "emergency"})
    
    assert isinstance(result, ActionResult)
    assert result.success is True
    assert "directions to emergency" in result.message
    assert "30 mins" in result.message
    assert "5 km" in result.message
    assert "map_image_path" in result.data["additional_info"]
    assert "google_maps_url" in result.data["additional_info"]

def test_execute_find_location(print_map_action, mock_gmaps, mock_requests):
    """Test executing find_location intent."""
    # Mock Google Places API response
    mock_gmaps.places.return_value = {
        'results': [{
            'geometry': {'location': {'lat': 37.7749, 'lng': -122.4194}},
            'place_id': 'test_place_id',
            'formatted_address': '123 Test St'
        }]
    }
    
    result = print_map_action.execute("find_location", {"destination": "cafeteria"})
    
    assert isinstance(result, ActionResult)
    assert result.success is True
    assert "floor 1" in result.message
    assert "West wing" in result.message
    assert "map_image_path" in result.data["additional_info"]
    assert "google_maps_url" in result.data["additional_info"]

def test_execute_unknown_location(print_map_action):
    """Test executing action with unknown location."""
    result = print_map_action.execute("show_map", {"destination": "unknown_place"})
    
    assert isinstance(result, ActionResult)
    assert result.success is False
    assert "don't have information" in result.message
    assert "error" not in result.data

def test_execute_places_api_error(print_map_action, mock_gmaps):
    """Test handling of Places API error."""
    mock_gmaps.places.return_value = {'results': []}
    
    result = print_map_action.execute("show_map", {"destination": "radiology"})
    
    assert isinstance(result, ActionResult)
    assert result.success is False
    assert "Could not find" in result.message
    assert "error" not in result.data

def test_execute_map_download_error(print_map_action, mock_gmaps, mock_requests):
    """Test handling of map download error."""
    # Mock Google Places API response
    mock_gmaps.places.return_value = {
        'results': [{
            'geometry': {'location': {'lat': 37.7749, 'lng': -122.4194}},
            'place_id': 'test_place_id',
            'formatted_address': '123 Test St'
        }]
    }
    
    # Mock request failure
    mock_requests.return_value.raise_for_status.side_effect = Exception("Failed to download map")
    
    result = print_map_action.execute("show_map", {"destination": "radiology"})
    
    assert isinstance(result, ActionResult)
    assert result.success is False
    assert result.message == "Failed to generate map image."
    assert result.data["error"] == "Failed to download map" 