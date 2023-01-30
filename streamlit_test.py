import streamlit as st

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer, EncoderDecoderModel, EarlyStoppingCallback)

#------------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast = True)
model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/Team_Project/Project3(translate)/best_model/klue-bert-dataset-15000")

#------------------------------------------------------------------------------------------

st.title("문어체/구어체 스타일 변환기")
from_text = st.text_input("번역할 글", "안녕하세요.")

btn_translate = st.button("번역하기")

source = st.selectbox("나의 언어 (또는 자동)", ('auto', "구어체", "문어체"))
destination = st.selectbox("무슨 언어로 번역할지", ("문어체", "구어체"))

if btn_translate:  # 버튼 누르면
    if not source or source == "auto":  # 나의 언어 선택을 안했거나, "auto"이면
        # 스타일 감지하기
        model.eval()
        model.cuda()
        embeddings = tokenizer(from_text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        embeddings = {k: v.cuda() for k, v in embeddings.items()}
        output = model(**embeddings)
        preds = output.logits.argmax(dim=-1)
        if preds == 0:
            source = '구어체'
        elif preds == 1:
            source = '문어체'

    if not destination:  # 무슨 언어로 번역할지 선택을 안했으면
        destination = "문어체"  # 기본은 문어체로 한다.

    if source == '구어체' or destination == '구어체':
        result =  '구어체 입니다.'
    elif source == '문어체' or destination == '문어체':
        result =  '문어체 입니다.'

    st.success(result)
    st.write(f"Translated from {source if source else preds} to {destination}")
