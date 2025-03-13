from PyInstaller.utils.hooks import collect_data_files
import spacy
import os

datas = collect_data_files('spacy')

# Add the model data directory
model_name = "en_core_web_md"  # Replace with your model
model_path = os.path.join(spacy.util.get_data_path(), model_name)
datas.append((model_path, model_name))