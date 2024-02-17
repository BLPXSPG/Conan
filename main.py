from extract import CharExtraction, SaveEmbedding, Evaluation
from config import Config
import openai

cfg = Config()

eva = Evaluation(cfg)
#eva.copy_folders_labelled()
#import os
#eva.evaluate_all_baselines()
#eva.get_all_stats()
#eva.count_character_number_all()

ext = CharExtraction(cfg)
ext.extract_all()
#ext.extract_all(language="english")
#ext.translate_all()
#ext.check_relation_language()
#ext.check_relation_language(language="english")
