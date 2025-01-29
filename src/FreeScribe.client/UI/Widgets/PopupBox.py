import tkinter as tk
from tkinter import Toplevel

class PopupBox:
    def __init__(self, 
        parent, 
        title="Message", 
        message="Message text", 
        button_text_1="OK", 
        button_text_2="Cancel", 
        button_1_callback=None, 
        button_2_callback=None):
        
        self.response = None
        self.dialog = Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x150")
        self.dialog.resizable(False, False)
        self.button_1_callback = button_1_callback
        self.button_2_callback = button_2_callback

        # Message label
        label = tk.Label(self.dialog, text=message, wraplength=250)
        label.pack(pady=20)

        # Button frame
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)

        # Buttons
        button_1 = tk.Button(button_frame, text=button_text_1, command=self.on_button_1)
        button_1.pack(side=tk.LEFT, padx=10)

        button_2 = tk.Button(button_frame, text=button_text_2, command=self.on_button_2)
        button_2.pack(side=tk.RIGHT, padx=10)

        # Modal behavior
        self.dialog.transient(parent)
        self.dialog.grab_set()
        parent.wait_window(self.dialog)

    def on_button_1(self):
        self.response = "button_1"
        self.button_1_callback()
        self.dialog.destroy()

    def on_button_2(self):
        self.response = "button_2"
        self.button_2_callback()
        self.dialog.destroy()


# Example usage
def main():
    root = tk.Tk()
    root.geometry("400x300")

    def show_custom_message_box():
        message_box = CustomMessageBox(
            root,
            title="Invalid Input",
            message="The input does not meet the requirements. Do you want to continue?",
            button_text_1="Continue",
            button_text_2="Cancel"
        )
        print(f"User response: {message_box.response}")

    # Button to show the custom message box
    show_message_button = tk.Button(root, text="Show Message Box", command=show_custom_message_box)
    show_message_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
