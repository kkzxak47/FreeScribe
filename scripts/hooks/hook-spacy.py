from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# Collect all spaCy data files
datas = collect_data_files('spacy')

# Collect all spaCy submodules
hiddenimports = collect_submodules('spacy')

# Add the en_core_web_md model data and its metadata
model_data = collect_data_files('en_core_web_md')
model_dist_info = collect_data_files('en_core_web_md-3.7.1.dist-info')

# Explicitly include __init__.py files
init_files = [
    ('en_core_web_md/__init__.py', 'en_core_web_md'),
]

datas += model_data + model_dist_info + init_files

# Add specific hidden imports for the model
hiddenimports += [
    'en_core_web_md',
] 