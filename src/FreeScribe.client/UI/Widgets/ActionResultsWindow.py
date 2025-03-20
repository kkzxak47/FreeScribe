"""
Window for displaying intent action results.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import webbrowser
from typing import Dict, Any, List

class ActionResultsWindow:
    """
    Window for displaying intent action results.
    
    This window appears to the right of the main window and shows
    a list of cards containing action results like maps and directions.
    """
    
    def __init__(self, parent: tk.Tk):
        """
        Initialize the action results window.
        
        :param parent: Parent window
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Action Results")
        self.window.geometry("400x600")
        self.window.resizable(True, True)
        
        # Position window to the right of parent
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        self.window.geometry(f"+{parent_x + parent.winfo_width()}+{parent_y}")
        
        # Create scrollable frame
        self.canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store image references to prevent garbage collection
        self.images = []
        
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a new action result card to the window.
        
        :param result: Action result data
        """
        # Create card frame
        card = ttk.Frame(self.scrollable_frame, style="Card.TFrame")
        card.pack(fill="x", padx=10, pady=5)
        
        # Add header
        header = ttk.Frame(card)
        header.pack(fill="x", padx=5, pady=5)
        
        icon = ttk.Label(header, text=result["ui"]["icon"])
        icon.pack(side="left", padx=5)
        
        title = ttk.Label(header, text=result["display_name"], style="CardTitle.TLabel")
        title.pack(side="left", padx=5)
        
        # Add message
        message = ttk.Label(card, text=result["message"], wraplength=350)
        message.pack(fill="x", padx=10, pady=5)
        
        # Add additional info if available
        if "additional_info" in result["data"]:
            info = result["data"]["additional_info"]
            
            # Add map image if available
            if "map_image_path" in info:
                try:
                    image = Image.open(info["map_image_path"])
                    image = image.resize((350, 350), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    self.images.append(photo)  # Prevent garbage collection
                    
                    map_label = ttk.Label(card, image=photo)
                    map_label.pack(pady=10)
                    
                    # Add Google Maps link
                    if "google_maps_url" in info:
                        link = ttk.Label(
                            card, 
                            text="Open in Google Maps", 
                            cursor="hand2",
                            foreground="blue"
                        )
                        link.pack(pady=5)
                        link.bind("<Button-1>", lambda e: webbrowser.open(info["google_maps_url"]))
                except Exception as e:
                    print(f"Error loading map image: {e}")
            
            # Add other info
            if "floor" in info:
                floor = ttk.Label(card, text=info["floor"])
                floor.pack(pady=2)
                
            if "wing" in info:
                wing = ttk.Label(card, text=info["wing"])
                wing.pack(pady=2)
                
            if "key_landmarks" in info:
                landmarks = ttk.Label(card, text="Key Landmarks:")
                landmarks.pack(pady=2)
                for landmark in info["key_landmarks"]:
                    lm = ttk.Label(card, text=f"• {landmark}")
                    lm.pack(pady=1)
                    
        # Add separator
        ttk.Separator(self.scrollable_frame).pack(fill="x", padx=10, pady=10)
        
    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Add multiple action results to the window.
        
        :param results: List of action results
        """
        for result in results:
            self.add_result(result)
            
    def clear(self) -> None:
        """Clear all results from the window."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.images.clear()
        
    def show(self) -> None:
        """Show the window."""
        self.window.deiconify()
        self.window.lift()
        
    def hide(self) -> None:
        """Hide the window."""
        self.window.withdraw() 