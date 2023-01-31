import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BartForConditionalGeneration,
)
from kiwipiepy import Kiwi

kiwi = Kiwi()

if 'tokenizer' not in st.session_state:
    classifier_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base', use_fast = True)
    converter_tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
    st.session_state.tokenizer = classifier_tokenizer, converter_tokenizer
else:
    classifier_tokenizer, converter_tokenizer = st.session_state.tokenizer

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()}, allow_output_mutation=True)
def get_classifier():
    model = AutoModelForSequenceClassification.from_pretrained('yangdk/bert-base-finetuned-spoken-written')
    model.eval()
    return model
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()}, allow_output_mutation=True)
def get_spoken_to_written_converter():
    model = BartForConditionalGeneration.from_pretrained('yangdk/kobart-base-v2-finetuned-spoken-to-written')
    model.eval()
    return model
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()}, allow_output_mutation=True)
def get_written_to_spoken_converter():
    model = BartForConditionalGeneration.from_pretrained('yangdk/kobart-base-v2-finetuned-written-to-spoken')
    model.eval()
    return model

classifier = get_classifier()
spoken_to_written_converter = get_spoken_to_written_converter()
written_to_spoken_converter = get_written_to_spoken_converter()

st.title("Spoken-Written Converter 🪄")

str_to_int = {
    'Spoken 🗣️': 0,
    'Written ✍️': 1 
}
goal = str_to_int[st.radio("Set Goal 👇", ('Spoken 🗣️', 'Written ✍️'))]

col1, col2 = st.columns(2)
with col1:
    input_text = st.text_input("Type your text here 👇")
    bnt = st.button('Convert 🪄')
outputs = []
with col2:
    if input_text and bnt:
        embeddings = classifier_tokenizer(input_text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        output = classifier(**embeddings)
        preds = output.logits.argmax(dim=-1)
        if preds == goal:
            st.success(f'No alerts')
        else:
            if preds == 0:
                for text in kiwi.split_into_sents(input_text):
                    tokenized = converter_tokenizer(text.text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
                    out = spoken_to_written_converter.generate(**tokenized, max_length=128)[0, 1:-1]
                    out = converter_tokenizer.decode(out, skip_special_tokens=True)
                    outputs.append(out)
                st.text_area("Written ✍️", value=" ".join(outputs))
                st.error(f'Your text is Spoken 🗣️ style')
            else:
                for text in kiwi.split_into_sents(input_text):
                    tokenized = converter_tokenizer(text.text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
                    out = written_to_spoken_converter.generate(**tokenized, max_length=128)[0, 1:-1]
                    out = converter_tokenizer.decode(out, skip_special_tokens=True)
                    outputs.append(out)
                st.text_area("Spoken 🗣️", value=" ".join(outputs))
                st.error(f'Your text is Written ✍️ style')
