from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# Collect all spaCy data files
datas = collect_data_files('spacy')

# Collect all spaCy submodules
hiddenimports = collect_submodules('spacy')

# Add the en_core_web_md model data and its metadata
model_data = collect_data_files('en_core_web_md')
model_dist_info = collect_data_files('en_core_web_md-3.7.1.dist-info')
datas += model_data + model_dist_info

# Add specific hidden imports for the model
hiddenimports += [
    'en_core_web_md',
    'en_core_web_md.language',
    'en_core_web_md.language.lex_attrs',
    'en_core_web_md.language.syntax_iterators',
    'en_core_web_md.language.tag_map',
    'en_core_web_md.language.tokenizer_exceptions',
    'en_core_web_md.language.wordnet',
    'en_core_web_md.language.lemmatizer',
    'en_core_web_md.language.stop_words',
    'en_core_web_md.language.punctuation',
    'en_core_web_md.language.char_classes',
    'en_core_web_md.language.lex_attrs',
    'en_core_web_md.language.syntax_iterators',
    'en_core_web_md.language.tag_map',
    'en_core_web_md.language.tokenizer_exceptions',
    'en_core_web_md.language.wordnet',
    'en_core_web_md.language.lemmatizer',
    'en_core_web_md.language.stop_words',
    'en_core_web_md.language.punctuation',
    'en_core_web_md.language.char_classes',
] 