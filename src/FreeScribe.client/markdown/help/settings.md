# Settings Documentation

## General Settings

### Show Welcome Message

**Description**: Display a welcome message on application startup.  
**Default**: `true`  
**Type**: boolean

### Show Scrub PHI

**Description**: Enable or disable PHI (Protected Health Information) scrubbing. This feature applies to local LLMs (Large Language Models) and private networks. It is used to remove potentially sensitive data before sending it to the model. For internet-facing endpoints, this feature is always enabled regardless of the setting.  
**Default**: `false`  
**Type**: boolean

---

## Speech-to-Text (Whisper) Settings

### Built-in Speech2Text

**Description**: Enable the built-in Speech-to-Text feature.  
**Default**: `true`  
**Type**: boolean

### Real-Time Speech-to-Text

**Description**: Enable real-time transcription of speech to text.  
**Default**: `true`  
**Type**: boolean

### Whisper Model

**Description**: Select the model size for Whisper transcription. Options include small, medium, and large models.  
**Default**: `medium`  
**Type**: dropdown

### Speech2Text (Whisper) Endpoint

**Description**: API endpoint for Whisper service. This endpoint processes audio files from the client. By default, it points to the local Whisper container provided by ClinicianFOCUS.  
**Default**: `https://localhost:2224/whisperaudio`  
**Type**: string

### Speech2Text (Whisper) API Key

**Description**: API key for accessing the Whisper endpoint.  
**Default**: (empty)  
**Type**: string

### Speech2Text Server Self-Signed Certificates

**Description**: Allow self-signed certificates for the Speech-to-Text server connection.  
**Default**: `false`  
**Type**: boolean

### Speech2Text (Whisper) Architecture

**Description**: Specify the hardware architecture used for Whisper Speech-to-Text processing. Options include `CUDA (Nvidia GPU)` and `CPU` available architectures.  
**Default**: `CPU`  
**Type**: dropdown

---

## AI Settings

### Builtin AI processing

**Description**: Enable or disable the use of a local large language model. This will run on your device if enabled.  
**Default**: `true`  
**Type**: boolean

### Builtin AI Architecture

**Description**: Choose the hardware architecture for running the local LLM. Options may include `CUDA (Nvidia GPU)` and `CPU`.  
**Default**: `CPU`  
**Type**: dropdown

### Builtin AI model processing

**Description**: Select the local LLM model to use.  
**Default**: `gemma-2-2b-it-Q8_0.gguf`  
**Type**: dropdown

### AI Server Endpoint

**Description**: Endpoint for the local AI model server.  
**Default**: `https://localhost:3333`  
**Type**: string

### AI Server Self-Signed Certificates

**Description**: Allow self-signed certificates for the AI model server connection.  
**Default**: `false`  
**Type**: boolean

### AI Server API Key

**Description**: API key for accessing the OpenAI API if a remote model is used.  
**Default**: `None`  
**Type**: string

---

## Advanced Settings

### Audio Processing Timeout

**Description**: Timeout in seconds for audio processing tasks.  
**Default**: `180`  
**Type**: integer

### Whisper Beam Size (Experimental)

**Description**: Set the beam search size for Whisper's transcription process.  
**Default**: `5`  
**Type**: integer

### Whisper CPU Thread Count (Experimental)

**Description**: Number of CPU threads allocated for Whisper processing.  
**Default**: `Auto (Detects based on machine)`  
**Type**: integer

### Use Whisper VAD Filter (Experimental)

**Description**: Enable the use of Voice Activity Detection (VAD) to improve transcription accuracy.  
**Default**: `false`  
**Type**: boolean

### Whisper Compute Type (Experimental)

**Description**: Specify the data type for Whisper's computations, e.g., `float16`. This may get force to a different compute type if it is incompatible with your system.
**Default**: `float16`  
**Type**: string

### Translate Speech to English Text

**Description**: Enable automatic translation of transcribed speech to English.  
**Default**: `false`  
**Type**: boolean

### Whisper Language Code

**Description**: Specify the language code for transcription. If left empty, the language is auto-detected.  
**Default**: `None (Auto Detect)`  
**Type**: string

### Use Pre-Screen AI Input

**Description**: Pre-screen input data before processing by the AI model.  
**Default**: `true`  
**Type**: boolean

<!-- ### Use best_of

**Description**: Enable the use of multiple result candidates and choose the best output.
**Default**: `false`
**Type**: boolean -->

### Temperature

**Description**: Control randomness in model output. Lower values produce more deterministic results.  
**Default**: `0.1`  
**Type**: float

### tfs

**Description**: Truncated frequency sampling parameter for controlling model output diversity.  
**Default**: `0.97`  
**Type**: float

### top_k

**Description**: Limit the number of top tokens considered during generation.  
**Default**: `30`  
**Type**: integer

### top_p

**Description**: Control the probability threshold for top-p (nucleus) sampling.  
**Default**: `0.4`  
**Type**: float

---

## Prompting Settings

### Note Generation Prompt

**Description**: Template prompt for generating SOAP notes based on conversations.  
**Default**: See predefined text in settings.  
**Type**: string

### Post Prompting Instructions

**Description**: Additional instructions on how the AI should structure generated notes, ensuring accuracy and proper formatting in SOAP format.  
**Default**: See predefined text in settings.  
**Type**: string
