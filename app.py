import streamlit as st
import pandas as pd
import json
import os
import nltk
from nltk import sent_tokenize

# Descargar el modelo punkt si no está disponible
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_txt_files(uploaded_files, segment_by_sentences):
    structured_data = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == 'text/plain':
            content = uploaded_file.read().decode('utf-8')
            file_name = uploaded_file.name
            if segment_by_sentences:
                sentences = sent_tokenize(content)
                for sentence in sentences:
                    structured_data.append({'filename': file_name, 'content': sentence})
            else:
                structured_data.append({'filename': file_name, 'content': content})
    return structured_data


def save_as_json(data, content_key, label_key, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    structured_output = [{content_key: item['content'], label_key: []} for item in data]
    json_path = os.path.join(output_dir, 'structured_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structured_output, f, indent=4, ensure_ascii=False)
    return json_path, structured_output


def save_as_jsonl(data, content_key, label_key, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, 'structured_data.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps({content_key: item['content'], label_key: []}, ensure_ascii=False)
            f.write(json_line + '\n')
    return jsonl_path


def save_as_csv(data, content_key, label_key, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame([{content_key: item['content'], label_key: ''} for item in data])
    csv_path = os.path.join(output_dir, 'structured_data.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    return csv_path, df


st.title('Conversor de TXT a JSON, JSONL o CSV')

st.write('## ¿Qué formato necesitas?')
st.markdown('---')
st.write('**JSON:** Formato estructurado ideal para análisis o procesamiento posterior. Cada archivo subido es almacenado como un objeto en una lista JSON.')
st.write('Ejemplo:')
st.code('[{"Texto": "Terrible customer service.", "Etiqueta": ["NEG"]}, {"Texto": "Excellent product.", "Etiqueta": ["POS"]}]')

st.write('**JSONL:** Formato similar a JSON pero con un objeto por línea. Ideal para procesamiento a gran escala o entrenamiento de modelos de aprendizaje automático.')
st.write('Ejemplo:')
st.code('{"Texto": "Terrible customer service.", "Etiqueta": ["NEG"]}\n{"Texto": "Excellent product.", "Etiqueta": ["POS"]}')

st.write('**CSV:** Formato tabular comúnmente utilizado para manipulación en hojas de cálculo o análisis en pandas.')
st.write('Ejemplo:')
st.code('Texto,Etiqueta\n"Terrible customer service.","NEG"\n"Excellent product.","POS"')
st.markdown('---')

uploaded_files = st.file_uploader('Sube tus archivos .txt', type='txt', accept_multiple_files=True)

if uploaded_files:
    segment_by_sentences = st.checkbox('Segmentar por oraciones')
    content_key = st.text_input('Nombre para el contenido (ej. "Texto")', value='content')
    label_key = st.text_input('Nombre para la etiqueta (ej. "Etiqueta")', value='label')
    file_name = st.text_input('Nombre del archivo a descargar (sin extensión)', value='structured_data')

    structured_data = process_txt_files(uploaded_files, segment_by_sentences)

    if st.button('Guardar como JSON'):
        json_path, structured_output = save_as_json(structured_data, content_key, label_key, 'output')
        st.success(f'Datos guardados como JSON en: {json_path}')
        st.download_button(label='Descargar JSON', data=json.dumps(structured_output, indent=4), file_name=f'{file_name}.json', mime='application/json')

    if st.button('Guardar como JSONL'):
        jsonl_path = save_as_jsonl(structured_data, content_key, label_key, 'output')
        st.success(f'Datos guardados como JSONL en: {jsonl_path}')
        jsonl_data = '\n'.join([json.dumps({content_key: item['content'], label_key: []}, ensure_ascii=False) for item in structured_data]).encode('utf-8')
        st.download_button(label='Descargar JSONL', data=jsonl_data, file_name=f'{file_name}.jsonl', mime='application/json')

    if st.button('Guardar como CSV'):
        csv_path, df = save_as_csv(structured_data, content_key, label_key, 'output')
        st.success(f'Datos guardados como CSV en: {csv_path}')
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label='Descargar CSV', data=csv_data, file_name=f'{file_name}.csv', mime='text/csv')
