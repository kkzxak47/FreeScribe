from tkinter import Toplevel, messagebox
import markdown as md
import tkinter as tk
from tkhtmlview import HTMLLabel
from utils.file_utils import get_file_path
from utils.utils import get_application_version
import re
from UI.ImageWindow import ImageWindow

class MarkdownWindow:
    """
    A class to display a Markdown file in a pop-up window with optional callback functionality.

    Parameters:
    -----------
    parent : widget
        The parent widget.
    title : str
        The title of the window.
    file_path : str
        The path to the Markdown file to be rendered.
    callback : function, optional
        A callback function to be called when the window is closed, with the state of the checkbox.
    """

    def __init__(self, parent, title, file_path, callback=None):
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                content = file.read()
            content = self._process_markdown_images(content)
            html_content = md.markdown(content, extensions=["extra", "smarty"])
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            messagebox.showerror("Error", "File not found")
            return
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding='cp1252') as file:
                    content = file.read()
                content = self._process_markdown_images(content)
                html_content = md.markdown(content, extensions=["extra", "smarty"])
            except Exception as e:
                print(f"Error reading file: {e}")
                messagebox.showerror("Error", "Error reading file")
                return

        self.parent = parent
        self.window = Toplevel(parent)
        self.window.title(title)
        self.window.transient(parent)
        self.window.grab_set()
        self.window.iconbitmap(get_file_path('assets', 'logo.ico'))

        self.parent_mousewheel_binding = parent.bind_all("<MouseWheel>")
        
        frame = tk.Frame(self.window)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.html_label = HTMLLabel(frame, html=html_content)
        self.html_label.pack(side="left", fill="both", expand=True)
        
        self.scrollbar = tk.Scrollbar(frame, orient="vertical", command=self.html_label.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.html_label.config(yscrollcommand=self.scrollbar.set)
        
        self.html_label.bind('<Enter>', self._bind_mousewheel)
        self.html_label.bind('<Leave>', self._unbind_mousewheel)
        
        footer_frame = tk.Frame(self.window, bg="lightgray")
        footer_frame.pack(side=tk.BOTTOM, fill="x", pady=5)
        version = get_application_version()

        if callback:
            var = tk.BooleanVar()
            check_button = tk.Checkbutton(footer_frame, text="Don't show this message again", 
                                          variable=var, bg="lightgray")
            check_button.grid(row=0, column=0, padx=5, pady=3)
            close_button = tk.Button(footer_frame, text="Close", 
                                     command=lambda: self._on_close(var, callback), width=6)
            close_button.grid(row=1, column=0, padx=5, pady=3)
        else:
            close_button = tk.Button(footer_frame, text="Close", 
                                     command=self._on_window_close, width=6)
            close_button.grid(row=0, column=0, padx=5, pady=3)
        
        version_label = tk.Label(footer_frame, text=f"Version: {version}", 
                                 pady=2, padx=5, relief="sunken", font=("Arial", 8))
        version_label.grid(row=0, column=1, sticky='e')
        
        footer_frame.grid_columnconfigure(0, weight=1)
        footer_frame.grid_columnconfigure(1, weight=0)
        
        # Create a new frame for the help button
        help_button_frame = tk.Frame(self.window)
        
        if title == "Welcome":
            help_button_frame.pack(side=tk.BOTTOM, fill="x", pady=5)
            help_button = tk.Button(help_button_frame, text="Click to View Interface Guide", 
                                    command=lambda: self._open_image_window())
            help_button.pack(side=tk.BOTTOM, padx=5, pady=3, anchor='center')
        
        self._adjust_window_size(self.html_label, self.scrollbar)
        self._display_to_center()
        self.window.lift()
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
    
    def _open_image_window(self):
        image_window = ImageWindow(self.parent, "Help Guide", get_file_path('assets', 'help.png'))
        image_window.window.transient(self.window)  # Set ImageWindow as transient to MarkdownWindow
        image_window.window.grab_set()              # Ensure ImageWindow gets focus
        image_window.window.focus_force()           # Force focus on ImageWindow
    
    def _process_markdown_images(self, content):
        """
        Process markdown image tags to add size constraints.
        """
        # Regular expression to find markdown image tags
        image_pattern = r'!$$(.*?)$$$$(.*?)$$'
        
        def replace_with_html(match):
            alt_text = match.group(1)
            image_path = match.group(2)
            return f'<img src="{image_path}" alt="{alt_text}" width="500" />'
        
        return re.sub(image_pattern, replace_with_html, content)
    
    def _bind_mousewheel(self, event):
        """Bind mousewheel to HTMLLabel and temporarily unbind parent's mousewheel"""
        if self.parent_mousewheel_binding:
            self.parent.unbind_all("<MouseWheel>")
        self.window.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _unbind_mousewheel(self, event):
        """Unbind mousewheel from HTMLLabel and restore parent's mousewheel binding"""
        self.window.unbind_all("<MouseWheel>")
        if self.parent_mousewheel_binding:
            self.parent.bind_all("<MouseWheel>", self.parent_mousewheel_binding)
        
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.html_label.yview_scroll(int(-1 * (event.delta/120)), "units")
        return "break"
    
    def _display_to_center(self):
        """Center the window on the screen"""
        # Get parent window dimensions and position
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        center_x = parent_x + (parent_width - width) // 2
        center_y = parent_y + (parent_height - height) // 2
        self.window.geometry(f"{width}x{height}+{center_x}+{center_y}")
    
    def _adjust_window_size(self, html_label, scrollbar):
        """
        Dynamically adjusts the window size based on the content, with constraints.

        Parameters:
        -----------
        html_label : HTMLLabel
            The label containing the rendered Markdown content.
        scrollbar : Scrollbar
            The scrollbar associated with the HTMLLabel.
        """
        self.window.update_idletasks()  # Ensure all widgets are rendered

        content_width = html_label.winfo_reqwidth() + scrollbar.winfo_reqwidth() + 20
        content_height = html_label.winfo_reqheight() + 20
        width = min(content_width, 900)
        height = min(content_height, 750)
        self.window.geometry(f"{width}x{height}")
    
    def _on_close(self, var, callback):
        """
        Handles the window close event with callback.

        Parameters:
        -----------
        var : BooleanVar
            The Tkinter BooleanVar associated with the checkbox.
        callback : function
            The callback function to be called with the state of the checkbox.
        """
        callback(var.get())
        self._on_window_close()
    
    def _on_window_close(self):
        """Clean up bindings and restore parent's mousewheel binding before closing"""
        self.window.unbind_all("<MouseWheel>")
        if self.parent_mousewheel_binding:
            self.parent.bind_all("<MouseWheel>", self.parent_mousewheel_binding)
        self.window.destroy()