from PyInstaller.utils.hooks import collect_data_files, copy_metadata

# Get the language model's data files
datas = collect_data_files('en_core_web_md')

# Explicitly add the __init__.py file
import os
import en_core_web_md
model_path = os.path.dirname(en_core_web_md.__file__)
init_file = os.path.join(model_path, '__init__.py')
datas.append((init_file, 'en_core_web_md'))

# Also include the package metadata
datas += copy_metadata('en_core_web_md')