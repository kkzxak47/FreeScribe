# Application lock class to prevent multiple instances of an app from running
import tkinter as tk
from tkinter import messagebox
import psutil  # For process management
import sys
import win32gui  # Windows GUI functions
import win32con  # Windows GUI constants  
import win32process  # Windows process management

class ApplicationLock:
    """
    Controls application instances to ensure only one is running.
    
    Args:
        app_name: Window title of the application
        app_task_manager_name: Process name as shown in Task Manager
    """
    def __init__(self, app_name, app_task_manager_name):
        self.app_name = app_name
        self.app_task_manager_name = app_task_manager_name
        self.root = None
        
    def get_running_instance_pid(self):
        """
        Finds PIDs of any running instances of the application.
        
        Returns:
            list: PIDs of running instances
        """
        possible_ids = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == f"{self.app_task_manager_name}":
                    possible_ids.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return possible_ids
        
    def kill_instance(self, pid):
        """
        Terminates specified process instance(s).
        
        Args:
            pid: Process ID (int) or list of PIDs to terminate
        
        Returns:
            bool: True if termination successful, False otherwise
        """
        try:
            if type(pid) == int:
                process = psutil.Process(pid)
                process.terminate()
                return True
            elif type(pid) == list:
                for pid in pid:
                    process = psutil.Process(pid)
                    process.terminate()
                return True
        except psutil.NoSuchProcess:
            return False
        return False
            
    def bring_to_front(self):
        """Brings existing application window to foreground"""
        hwnd = win32gui.FindWindow(None, self.app_name)
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            
    def show_instance_dialog(self):
        """
        Shows dialog when another instance is detected.
        Allows user to close existing instance or cancel.
        
        Returns:
            bool: True if existing instance continues, False if terminated
        """
        dialog = tk.Tk()
        dialog.title("FreeScribe Instance")
        dialog.geometry("300x150")

        # Force dialog to front
        dialog.attributes("-topmost", True)
        dialog.lift()
        dialog.focus_force()
        
        pid = self.get_running_instance_pid()
        
        return_status = True

        label = tk.Label(dialog, text="Another instance of FreeScribe is already running.\nWhat would you like to do?")
        label.pack(pady=20)
        
        def handle_kill():
            """Handles clicking 'Close Existing Instance' button"""
            nonlocal return_status
            if self.kill_instance(pid):
                dialog.destroy()
                return_status = False
            else:
                messagebox.showerror("Error", "Failed to terminate existing instance")
                dialog.destroy()
                return_status = True
        
        def handle_cancel():
            """Handles clicking 'Cancel' button"""
            nonlocal return_status
            dialog.destroy()
            self.bring_to_front()
            return_status = True

        
        tk.Button(dialog, text="Close Existing Instance", command=handle_kill).pack(padx=5, pady=5)
        tk.Button(dialog, text="Cancel", command=handle_cancel).pack(padx=5, pady=2)
        
        dialog.mainloop()

        return return_status
        
    def run(self):
        """
        Main entry point to check for existing instances.
        
        Returns:
            bool: True if existing instance continues, False if none exists or terminated
        """
        if self.get_running_instance_pid():
            return self.show_instance_dialog()
        else:
            return False