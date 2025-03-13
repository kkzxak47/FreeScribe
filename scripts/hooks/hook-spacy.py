from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all spaCy data files
datas = collect_data_files('spacy')

# Collect all spaCy submodules
hiddenimports = collect_submodules('spacy')

# Add the en_core_web_md model data
datas += collect_data_files('en_core_web_md') 