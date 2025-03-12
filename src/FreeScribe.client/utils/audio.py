"""
This software is released under the AGPL-3.0 license
Copyright (c) 2023-2025 Braedon Hendy

Further updates and packaging added in 2024-2025 through the ClinicianFOCUS initiative, 
a collaboration with Dr. Braedon Hendy and Conestoga College Institute of Applied 
Learning and Technology as part of the CNERG+ applied research project, 
Unburdening Primary Healthcare: An Open-Source AI Clinician Partner Platform". 
Prof. Michael Yingbull (PI), Dr. Braedon Hendy (Partner), 
and Research Students (Software Developers) - 
Alex Simko, Pemba Sherpa, Naitik Patel, Yogesh Kumar and Xun Zhong.
"""

import numpy as np

def pad_audio_chunk(chunk, pad_seconds=0.5):
    """
    Pad an audio chunk with silence at the beginning and end.
    
    Parameters
    ----------
    chunk : np.ndarray
        The audio chunk to pad.
    pad_seconds : float
        The number of seconds to pad the chunk with.

    Returns
    -------
    np.ndarray
        The padded audio chunk.
    """

    # Calculate how many chunks make up half a second
    pad_chunk_leng = int(pad_seconds * RATE / CHUNK)

    # Create half a second of silence (all zeros)
    silent_chunk = np.zeros(CHUNK, dtype=np.int16).tobytes()

    # Create arrays of silent chunks
    silence_start = [silent_chunk] * half_second_chunks
    silence_end = [silent_chunk] * half_second_chunks

    # Add silence to the beginning and end of chunk
    padded_chunk = silence_start + chunk + silence_end

    return padded_chunk