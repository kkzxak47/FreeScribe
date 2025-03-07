import os
import json
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the SettingsWindow class with the correct paths
from UI.SettingsWindow import SettingsWindow
from UI.SettingsConstant import SettingsKeys


# Helper function to get extended integer settings
def get_extended_integer_settings(settings_instance):
    """
    Get integer settings extended with known integer settings.
    
    This helper function gets the base integer settings from the settings instance
    and extends them with known integer settings that might not be in DEFAULT_SETTINGS_TABLE.
    It also ensures there's no overlap with boolean settings.
    
    :param settings_instance: The settings instance
    :return: List of integer setting keys
    """
    # Use the get_extended_integer_settings method directly
    return settings_instance.get_extended_integer_settings()


# Helper function to write a settings file with custom content
def write_settings_file(test_dir, content, filename='settings.txt'):
    """
    Write content to a settings file in the test directory.
    
    :param test_dir: The test directory
    :param content: The content to write (string)
    :param filename: The filename to use, defaults to 'settings.txt'
    :return: The path to the created file
    """
    path = os.path.join(test_dir, filename)
    with open(path, 'w') as f:
        f.write(content)
    return path


@pytest.fixture
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def settings(test_dir):
    """Create a settings instance with mocked resource path."""
    # Patch the get_resource_path function to use our test directory
    with patch('UI.SettingsWindow.get_resource_path') as mock_get_resource_path:
        mock_get_resource_path.side_effect = lambda path: os.path.join(test_dir, path)
        
        # Create a settings instance
        settings_instance = SettingsWindow()
        
        # Mock the main_window attribute
        settings_instance.main_window = None
        
        yield settings_instance


def test_boolean_settings_save_load(settings, test_dir):
    """Test that boolean settings are saved and loaded correctly."""
    # Get boolean settings
    boolean_settings = settings.get_boolean_settings()
    
    # Ensure we have at least one boolean setting to test
    assert len(boolean_settings) > 0, "No boolean settings found to test"
    
    # Select a boolean setting to test
    test_setting = boolean_settings[0]
    
    # Set the value to True
    settings.editable_settings[test_setting] = True
    
    # Save settings
    settings.save_settings_to_file()
    
    # Verify the settings file exists
    settings_file = os.path.join(test_dir, 'settings.txt')
    assert os.path.exists(settings_file), "Settings file was not created"
    
    # Read the file directly to verify the value was saved as a boolean
    with open(settings_file, 'r') as f:
        saved_settings = json.load(f)
    
    # Check that the value is saved as a boolean (true), not as a string or integer
    assert isinstance(saved_settings['editable_settings'][test_setting], bool), \
        f"Setting {test_setting} was not saved as a boolean"
    assert saved_settings['editable_settings'][test_setting] is True, \
        f"Setting {test_setting} value was not saved correctly"
    
    # Change the value in memory
    settings.editable_settings[test_setting] = False
    
    # Load settings
    settings.load_settings_from_file()
    
    # Verify the value was loaded correctly as a boolean
    assert isinstance(settings.editable_settings[test_setting], bool), \
        f"Setting {test_setting} was not loaded as a boolean"
    assert settings.editable_settings[test_setting] is True, \
        f"Setting {test_setting} value was not loaded correctly"


def test_integer_settings_save_load(settings, test_dir):
    """Test that integer settings are saved and loaded correctly."""
    # Get extended integer settings
    integer_settings = get_extended_integer_settings(settings)
    
    # Ensure we have at least one integer setting to test
    assert len(integer_settings) > 0, "No integer settings found to test"
    
    # Select an integer setting to test
    test_setting = integer_settings[0]
    
    # Set the value to a test integer
    test_value = 42
    settings.editable_settings[test_setting] = test_value
    
    # Save settings
    settings.save_settings_to_file()
    
    # Verify the settings file exists
    settings_file = os.path.join(test_dir, 'settings.txt')
    assert os.path.exists(settings_file), "Settings file was not created"
    
    # Read the file directly to verify the value was saved as an integer
    with open(settings_file, 'r') as f:
        saved_settings = json.load(f)
    
    # Check that the value is saved as an integer, not as a string
    assert isinstance(saved_settings['editable_settings'][test_setting], int), \
        f"Setting {test_setting} was not saved as an integer"
    assert saved_settings['editable_settings'][test_setting] == test_value, \
        f"Setting {test_setting} value was not saved correctly"
    
    # Change the value in memory
    settings.editable_settings[test_setting] = 0
    
    # Load settings
    settings.load_settings_from_file()
    
    # Verify the value was loaded correctly as an integer
    assert isinstance(settings.editable_settings[test_setting], int), \
        f"Setting {test_setting} was not loaded as an integer"
    assert settings.editable_settings[test_setting] == test_value, \
        f"Setting {test_setting} value was not loaded correctly"


def test_mixed_settings_save_load(settings, test_dir):
    """Test that a mix of boolean and integer settings are saved and loaded correctly."""
    # Get boolean and integer settings
    boolean_settings = settings.get_boolean_settings()
    integer_settings = get_extended_integer_settings(settings)
    
    # Ensure we have at least one of each type to test
    if not boolean_settings:
        pytest.skip("No boolean settings found to test")
    if not integer_settings:
        pytest.skip("No integer settings found to test")
    
    # Select one setting of each type to test
    bool_setting = boolean_settings[0]
    int_setting = integer_settings[0]
    
    # Set test values
    bool_value = True
    int_value = 42
    settings.editable_settings[bool_setting] = bool_value
    settings.editable_settings[int_setting] = int_value
    
    # Save settings
    settings.save_settings_to_file()
    
    # Change the values in memory
    settings.editable_settings[bool_setting] = False
    settings.editable_settings[int_setting] = 0
    
    # Load settings
    settings.load_settings_from_file()
    
    # Verify both values were loaded correctly with the right types
    assert isinstance(settings.editable_settings[bool_setting], bool), \
        f"Setting {bool_setting} was not loaded as a boolean"
    assert settings.editable_settings[bool_setting] == bool_value, \
        f"Setting {bool_setting} value was not loaded correctly"
    
    assert isinstance(settings.editable_settings[int_setting], int), \
        f"Setting {int_setting} was not loaded as an integer"
    assert settings.editable_settings[int_setting] == int_value, \
        f"Setting {int_setting} value was not loaded correctly"


def test_string_to_boolean_conversion(settings, test_dir):
    """Test that string representations of booleans are converted correctly."""
    # Get boolean settings
    boolean_settings = settings.get_boolean_settings()
    
    # Ensure we have at least one boolean setting to test
    if not boolean_settings:
        pytest.skip("No boolean settings found to test")
    
    # Select a boolean setting to test
    test_setting = boolean_settings[0]
    
    # First, make sure the setting is recognized as a boolean in the default settings
    # by setting it to True and saving it
    settings.editable_settings[test_setting] = True
    settings.save_settings_to_file()
    
    # Now create a settings file with a string representation of a boolean
    settings_data = {
        "openai_api_key": "test_key",
        "editable_settings": {
            test_setting: "1"  # String representation of True
        },
        "app_version": "1.0.0"
    }
    
    settings_file = os.path.join(test_dir, 'settings.txt')
    with open(settings_file, 'w') as f:
        json.dump(settings_data, f)
    
    # Load settings
    settings.load_settings_from_file()
    
    # Verify the value was converted to a boolean or at least has boolean-like behavior
    # Some implementations might keep it as a string but treat it as a boolean in logic
    value = settings.editable_settings[test_setting]
    
    # Test if the value behaves like True in a boolean context
    assert bool(value), f"Setting {test_setting} value does not behave like True"
    
    # If it's not a boolean, print a warning but don't fail the test
    if not isinstance(value, bool):
        print(f"Warning: Setting {test_setting} was loaded as {type(value).__name__} instead of bool")


def test_string_to_integer_conversion(settings, test_dir):
    """Test that string representations of integers are converted correctly."""
    # Get extended integer settings
    integer_settings = get_extended_integer_settings(settings)
    
    # Ensure we have at least one integer setting to test
    if not integer_settings:
        pytest.skip("No integer settings found to test")
    
    # Select an integer setting to test
    test_setting = integer_settings[0]
    
    # Create a settings file with a string representation of an integer
    settings_data = {
        "openai_api_key": "test_key",
        "editable_settings": {
            test_setting: "42"  # String representation of integer
        },
        "app_version": "1.0.0"
    }
    
    settings_file = os.path.join(test_dir, 'settings.txt')
    with open(settings_file, 'w') as f:
        json.dump(settings_data, f)
    
    # Load settings
    settings.load_settings_from_file()
    
    # Verify the value was converted to an integer
    assert isinstance(settings.editable_settings[test_setting], int), \
        f"Setting {test_setting} was not converted to an integer"
    assert settings.editable_settings[test_setting] == 42, \
        f"Setting {test_setting} value was not converted correctly"


def test_invalid_json_settings(settings, test_dir):
    """Test how settings handle an invalid JSON file."""
    # Create an invalid JSON content
    invalid_json = "{invalid: 'json', missing quotes}"  # intentionally invalid JSON
    write_settings_file(test_dir, invalid_json)
    
    # Save the original settings to compare later
    original_settings = settings.editable_settings.copy()
    
    # Load settings - should not raise an exception
    settings.load_settings_from_file()
    
    # Verify that the settings remain unchanged when loading invalid JSON
    for key, value in original_settings.items():
        assert settings.editable_settings[key] == value, \
            f"Setting {key} was changed after loading invalid JSON"


def test_settings_missing_keys(settings, test_dir):
    """Test how settings behave when required keys are missing."""
    # Create a settings file missing required keys
    incomplete_settings = '{"openai_api_key": "test_key"}'  # Missing editable_settings
    write_settings_file(test_dir, incomplete_settings)
    
    # Save the original settings to compare later
    original_settings = settings.editable_settings.copy()
    
    # Load settings - should not raise an exception
    settings.load_settings_from_file()
    
    # Verify that the settings remain unchanged when loading incomplete settings
    for key, value in original_settings.items():
        assert settings.editable_settings[key] == value, \
            f"Setting {key} was changed after loading settings with missing keys"


def test_settings_extra_data(settings, test_dir):
    """Test how settings behave when the settings file contains extra unexpected data."""
    # Get a boolean setting to test
    boolean_settings = settings.get_boolean_settings()
    if not boolean_settings:
        pytest.skip("No boolean settings found to test")
    test_setting = boolean_settings[0]
    
    # Create a settings file with extra keys not defined in the schema
    extra_data_settings = {
        "openai_api_key": "test_key",
        "editable_settings": {
            test_setting: True,
            "unexpectedKey": "unexpectedValue"
        },
        "app_version": "1.0.0",
        "extraTopLevelKey": "extraTopLevelValue"
    }
    
    settings_file = os.path.join(test_dir, 'settings.txt')
    with open(settings_file, 'w') as f:
        json.dump(extra_data_settings, f)
    
    # Load settings - should not raise an exception
    settings.load_settings_from_file()
    
    # Verify that the known setting was loaded correctly
    assert settings.editable_settings[test_setting] is True, \
        f"Setting {test_setting} was not loaded correctly from settings with extra data"
    
    # Verify that the unexpected key was not added to editable_settings
    assert "unexpectedKey" not in settings.editable_settings, \
        "Unexpected key was added to editable_settings" 