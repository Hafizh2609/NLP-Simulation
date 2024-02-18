from liqfit.pipeline import ZeroShotClassificationPipeline
from liqfit.models import T5ForZeroShotClassification
from transformers import T5Tokenizer
import streamlit as st
import time

model = T5ForZeroShotClassification.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')
tokenizer = T5Tokenizer.from_pretrained('knowledgator/comprehend_it-multilingual-t5-base')
classifier = ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer,ypothesis_template = '{}', encoder_decoder = True)
st.title('Natural Language Project')
st.markdown('Hafizh Zaki Prasetyo Adi|hafizhzaki6661@gmail.com|https://www.linkedin.com/in/hafizhzpa/')
part=st.sidebar.radio("project",["sentimen", "emosi", "label khusus"],captions = ["menentukan label sentimen", "menentukan label emosi", "klasifikasi berdasarkan label yang ditentukan"])
text = st.text_input('text', 'Saya sudah menggunakan produk ini selama sebulan dan saya sangat puas dengan hasilnya')
multiclass = st.checkbox('Izinkan multi label')
if part=='label khusus':
    start=time.time()
    label = st.text_input('label', 'positive,negative,neutral')
    if st.button('run'):
        candidate_labels = label.split(',')
        result=classifier(text, candidate_labels, multi_label=multiclass)
        if not multiclass:
            st.text(f"label:{result['labels'][0]}")
            st.text(f"skor:{result['scores'][0]}")
        else:
            bool_score=[score>0.5 for score in result['scores']]
            st.text(f"label:{','.join([label for i,label in enumerate(result['labels']) if bool_score[i]])}")
            st.text(f"skor:{','.join([skor for i,skor in enumerate(result['scores']) if bool_score[i]])}")
        st.text(f"waktu:{time.time()-start}")
if part=='sentimen':
    start=time.time()
    if st.button('run'):
        candidate_labels = ["positive','negative','neutral"]
        result=classifier(text, candidate_labels, multi_label=multiclass)
        if not multiclass:
            st.text(f"label:{result['labels'][0]}")
            st.text(f"skor:{result['scores'][0]}")
        else:
            bool_score=[score>0.5 for score in result['scores']]
            st.text(f"label:{','.join([label for i,label in enumerate(result['labels']) if bool_score[i]])}")
            st.text(f"skor:{','.join([skor for i,skor in enumerate(result['scores']) if bool_score[i]])}")
        st.text(f"waktu:{time.time()-start}")
if part=='emotion':
    start=time.time()
    if st.button('run'):
        candidate_labels = ["bahagia", "sedih", "takut", "marah", "antisipasi", "terkejut", "jijik","percaya"]
        result=classifier(text, candidate_labels, multi_label=multiclass)
        if not multiclass:
            st.text(f"label:{result['labels'][0]}")
            st.text(f"skor:{result['scores'][0]}")
        else:
            bool_score=[score>0.5 for score in result['scores']]
            st.text(f"label:{','.join([label for i,label in enumerate(result['labels']) if bool_score[i]])}")
            st.text(f"skor:{','.join([skor for i,skor in enumerate(result['scores']) if bool_score[i]])}")
        st.text(f"waktu:{time.time()-start}")