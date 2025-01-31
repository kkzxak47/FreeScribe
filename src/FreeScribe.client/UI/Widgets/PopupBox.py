import tkinter as tk
from tkinter import Toplevel

class PopupBox:
    """
    A class to create a popup dialog box with a customizable message and two buttons.

    :param parent: The parent window for the popup dialog.
    :param title: The title of the popup window (default: "Message").
    :param message: The message displayed in the popup (default: "Message text").
    :param button_text_1: The text for the first button (default: "OK").
    :param button_text_2: The text for the second button (default: "Cancel").
    :param button_1_callback: Callback function for the first button (default: None).
    :param button_2_callback: Callback function for the second button (default: None).
    """

    def __init__(self, 
                 parent, 
                 title="Message", 
                 message="Message text", 
                 button_text_1="OK", 
                 button_text_2="Cancel", 
                 button_1_callback=None, 
                 button_2_callback=None):
        """
        Initialize the PopupBox instance and create the dialog window.

        :param parent: The parent widget for the dialog.
        :param title: The title of the dialog window.
        :param message: The message to be displayed in the dialog.
        :param button_text_1: The text label for the first button.
        :param button_text_2: The text label for the second button.
        :param button_1_callback: Optional callback function for the first button.
        :param button_2_callback: Optional callback function for the second button.
        """
        self.response = None  # Stores the response indicating which button was clicked
        # Create a top-level window for the popup
        self.dialog = Toplevel(parent)
        # Set the window title
        self.dialog.title(title)
        # Set the size of the window
        self.dialog.geometry("300x150")
        # Disable window resizing
        self.dialog.resizable(False, False)

        # Create and pack the message label
        label = tk.Label(self.dialog, text=message, wraplength=250)
        label.pack(pady=20)

        # Create and pack a frame to hold the buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)

        # Create and pack the first button
        button_1 = tk.Button(button_frame, text=button_text_1, command=self.on_button_1)
        button_1.pack(side=tk.LEFT, padx=10)

        # Create and pack the second button
        button_2 = tk.Button(button_frame, text=button_text_2, command=self.on_button_2)
        button_2.pack(side=tk.RIGHT, padx=10)

        # Configure the dialog as a modal window
        # Make the dialog appear on top of the parent window
        self.dialog.transient(parent)
        # Prevent interaction with other windows until this dialog is closed
        self.dialog.grab_set()
        # Wait until the dialog window is closed
        parent.wait_window(self.dialog)

    def on_button_1(self):
        """
        Handle the event when the first button is clicked.
        Sets the response to 'button_1' and closes the dialog.
        """
        self.response = "button_1"
        self.dialog.destroy()

    def on_button_2(self):
        """
        Handle the event when the second button is clicked.
        Sets the response to 'button_2' and closes the dialog.
        """
        self.response = "button_2"
        self.dialog.destroy()
