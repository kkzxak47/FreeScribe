import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyaudio
import numpy as np
from utils.file_utils import get_file_path
from UI.Widgets.MicrophoneSelector import MicrophoneState

class MicrophoneTestFrame:
    def __init__(self, history_frame, p):
        self.root = history_frame
        self.p = p
        
        # Configure history frame grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=4)
        self.root.grid_rowconfigure(1, weight=1)

        # Create timestamp listbox
        self.timestamp_listbox = tk.Listbox(self.root, height=20)
        self.timestamp_listbox.grid(row=0, column=0, sticky='nsew')
        self.timestamp_listbox.insert(tk.END, "Temporary Note History")
        self.timestamp_listbox.config(fg='grey')

        # Create frame for mic test
        self.frame = ttk.Frame(self.root)
        self.frame.grid(row=1, column=0, sticky='nsew', padx=0, pady=(5, 0))
        
        # Initialize microphone list and settings
        self.initialize_microphones()
        
        # Create mic test UI
        self.create_mic_test_ui()
        
        # Start volume meter updates
        self.update_volume_meter()

    def initialize_microphones(self):
        self.mic_list = []
        try:
            default_input_info = self.p.get_default_input_device_info()
            self.default_input_index = default_input_info['index']
        except IOError:
            self.default_input_index = None

        device_count = self.p.get_device_count()
        for i in range(device_count):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                device_name = device_info['name']
                excluded_names = ["Stereo Mix", "Loopback", "Virtual", "Output", 
                                "Wave Out", "What U Hear", "Aux", "Port", "Mix"]
                if not any(excluded_name.lower() in device_name.lower() 
                          for excluded_name in excluded_names):
                    self.mic_list.append((i, device_name))

        seen_names = set()
        unique_mics = []
        for device_index, device_name in self.mic_list:
            if device_name not in seen_names:
                unique_mics.append((device_index, device_name))
                seen_names.add(device_name)
        self.mic_list = unique_mics

        self.default_selection_index = 0
        for idx, (device_index, device_name) in enumerate(self.mic_list):
            if device_index == self.default_input_index:
                self.default_selection_index = idx
                break

    def create_mic_test_ui(self):
        # # Create a title label
        # title_label = ttk.Label(self.frame, text="Microphone Test", font=('Helvetica', 9, 'bold'))
        # title_label.pack(pady=(0, 5))

        # Frame for dropdown
        dropdown_frame = ttk.Frame(self.frame)
        dropdown_frame.pack(fill=tk.X, pady=(0, 5))

        # Create a container frame for center alignment
        center_frame = ttk.Frame(dropdown_frame)
        center_frame.pack(expand=True)

        # Create styles for all elements
        style = ttk.Style()
        style.configure('Mic.TCombobox', padding=(5, 5, 5, 5))
        style.configure('Green.TFrame', background='#2ecc71')
        style.configure('Yellow.TFrame', background='#f1c40f')
        style.configure('Red.TFrame', background='#e74c3c')
        style.configure('Inactive.TFrame', background='#95a5a6')

        # Dropdown for microphone selection
        mic_options = [f"{name}" for _, name in self.mic_list]
        self.mic_dropdown = ttk.Combobox(
            center_frame, 
            values=mic_options, 
            state='readonly', 
            width=30,
            style='Mic.TCombobox'
        )
        self.mic_dropdown.pack(pady=(0, 5))
        self.mic_dropdown.current(self.default_selection_index)
        
        # Bind selection change to save immediately
        self.mic_dropdown.bind('<<ComboboxSelected>>', self.on_mic_change)

        # Volume meter container
        meter_frame = ttk.Frame(self.frame)
        meter_frame.pack(fill=tk.X, pady=(0, 5))

        # Try to load mic icon
        try:
            mic_icon = Image.open(get_file_path('assets', 'mic_icon.png'))
            mic_icon = mic_icon.resize((24, 24))
            self.mic_photo = ImageTk.PhotoImage(mic_icon)
            mic_icon_label = ttk.Label(meter_frame, image=self.mic_photo)
            mic_icon_label.pack(side=tk.LEFT, padx=(0, 10))
        except Exception as e:
            print(f"Error loading microphone icon: {e}")

        # Create volume meter segments
        self.segments_frame = ttk.Frame(meter_frame)
        self.segments_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create segments
        self.SEGMENT_COUNT = 20
        self.segments = []
        for i in range(self.SEGMENT_COUNT):
            segment = ttk.Frame(self.segments_frame, width=10, height=20)
            segment.pack(side=tk.LEFT, padx=1)
            segment.pack_propagate(False)
            self.segments.append(segment)

    def on_mic_change(self, event):
        # Save the selection immediately
        selected_name = self.mic_dropdown.get()
        selected_index = None
        for device_index, device_name in self.mic_list:
            if device_name == selected_name:
                selected_index = device_index
                break
        if selected_index is not None:
            MicrophoneState.SELECTED_MICROPHONE_INDEX = selected_index

        # Reset volume meter
        for segment in self.segments:
            segment.configure(style='Inactive.TFrame')

    def update_volume_meter(self):
        selected_name = self.mic_dropdown.get()
        selected_index = None
        for device_index, device_name in self.mic_list:
            if device_name == selected_name:
                selected_index = device_index
                break

        try:
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                input_device_index=selected_index
            )
            
            data = stream.read(1024, exception_on_overflow=False)
            stream.stop_stream()
            stream.close()

            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))
            
            if np.isnan(rms) or rms <= 0:
                volume = 0
            else:
                scaling_factor = 500
                volume = min(max(int((rms / 32768) * scaling_factor), 0), 100)

            # Update segments
            active_segments = int((volume / 100) * self.SEGMENT_COUNT)
            for i, segment in enumerate(self.segments):
                if i < active_segments:
                    if i < self.SEGMENT_COUNT * 0.6:
                        segment.configure(style='Green.TFrame')
                    elif i < self.SEGMENT_COUNT * 0.8:
                        segment.configure(style='Yellow.TFrame')
                    else:
                        segment.configure(style='Red.TFrame')
                else:
                    segment.configure(style='Inactive.TFrame')

        except Exception as e:
            print(f"Error in update_volume_meter: {e}")
            for segment in self.segments:
                segment.configure(style='Inactive.TFrame')

        self.frame.after(100, self.update_volume_meter)

    def bind_listbox_select(self, callback):
        self.timestamp_listbox.bind('<<ListboxSelect>>', callback)