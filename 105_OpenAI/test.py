from transformers import T5Tokenizer, T5EncoderModel, T5Config
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

MAX_LENGTH = 256
DEFAULT_T5_NAME = 'google/t5-v1_1-base'

#tokenizer = T5Tokenizer.from_pretrained(DEFAULT_T5_NAME, model_max_length=MAX_LENGTH)
T5Config.from_pretrained('google/t5-v1_1-base')

