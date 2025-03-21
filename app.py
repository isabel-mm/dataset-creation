import streamlit as st
import pandas as pd
import json
import os
import nltk
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from nltk.tokenize import PunktSentenceTokenizer
import urllib.request


# Descargar el modelo punkt desde el GitHub oficial de NLTK
PUNKT_URL = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip'
nltk_data_path = './nltk_data'

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
    urllib.request.urlretrieve(PUNKT_URL, os.path.join(nltk_data_path, 'punkt.zip'))
    import zipfile
    with zipfile.ZipFile(os.path.join(nltk_data_path, 'punkt.zip'), 'r') as zip_ref:
        zip_ref.extractall(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Crear un tokenizer personalizado
punkt_tokenizer = PunktSentenceTokenizer()

def process_txt_files(uploaded_files, segment_by_sentences):
    structured_data = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode('utf-8')
        file_name = uploaded_file.name
        if segment_by_sentences:
            sentences = punkt_tokenizer.tokenize(content)
            for sentence in sentences:
                structured_data.append({'filename': file_name, 'content': sentence})
        else:
            structured_data.append({'filename': file_name, 'content': content})
    return structured_data


def process_xml_files(uploaded_files):
    structured_data = []
    for uploaded_file in uploaded_files:
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        file_name = uploaded_file.name
        text_content = ET.tostring(root, encoding='utf-8', method='text').decode('utf-8')
        structured_data.append({'filename': file_name, 'content': text_content})
    return structured_data


def save_as_xml(data, content_key, label_keys):
    root = ET.Element('data')
    for item in data:
        entry = ET.SubElement(root, 'entry')
        content_element = ET.SubElement(entry, content_key)
        content_element.text = item['content']
        for key in label_keys:
            label_element = ET.SubElement(entry, key)
            label_element.text = None  # Cambiar '' por None para tener etiquetas vacías correctamente.
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = parseString(xml_str)
    return dom.toprettyxml(indent='  ')


def save_as_json(data, content_key, label_keys, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    structured_output = [{content_key: item['content'], **{key: [] for key in label_keys}} for item in data]
    return structured_output


def save_as_csv(data, content_key, label_keys, output_dir):
    columns = [content_key] + label_keys
    df = pd.DataFrame([{content_key: item['content'], **{key: '' for key in label_keys}} for item in data], columns=columns)
    return df


def save_as_jsonl(data, content_key, label_keys, output_dir):
    jsonl_data = '\n'.join([json.dumps({content_key: item['content'], **{key: [] for key in label_keys}}, ensure_ascii=False) for item in data])
    return jsonl_data


st.title('Conversor de TXT/XML a JSON, JSONL, CSV o XML')

st.write('## ¿Qué formato necesitas?')
st.markdown('---')

# Ejemplos de formatos
st.write('**JSON:**')
st.code('[{"Texto": "Terrible customer service.", "Etiqueta": ["NEG"]}, {"Texto": "Excellent product.", "Etiqueta": ["POS"]}]')

st.write('**JSONL:**')
st.code('{"Texto": "Terrible customer service.", "Etiqueta": ["NEG"]}\n{"Texto": "Excellent product.", "Etiqueta": ["POS"]}')

st.write('**CSV:**')
st.code('Texto,Etiqueta\n"Terrible customer service.","NEG"\n"Excellent product.","POS"')

st.write('**XML:**')
st.code('<data><entry><Texto>Terrible customer service.</Texto><Etiqueta>NEG</Etiqueta></entry></data>')

st.markdown('---')

uploaded_files = st.file_uploader('Sube tus archivos .txt o .xml', type=['txt', 'xml'], accept_multiple_files=True)

if uploaded_files:
    segment_by_sentences = st.checkbox('Segmentar por oraciones (solo para archivos .txt)')
    content_key = st.text_input('Nombre para el contenido (ej. "Texto")', value='content')
    labels_input = st.text_input('Nombres de las etiquetas separados por comas (ej. "Etiqueta1, Etiqueta2")')
    label_keys = [label.strip() for label in labels_input.split(',')] if labels_input else ['label']
    file_name = st.text_input('Nombre del archivo a descargar (sin extensión)', value='structured_data')

    txt_files = [file for file in uploaded_files if file.type == 'text/plain']
    xml_files = [file for file in uploaded_files if file.type == 'text/xml']

    structured_data = process_txt_files(txt_files, segment_by_sentences) + process_xml_files(xml_files)

    if st.button('Guardar como JSON'):
        json_data = save_as_json(structured_data, content_key, label_keys, 'output')
        st.download_button(label='Descargar JSON', data=json.dumps(json_data, indent=4), file_name=f'{file_name}.json', mime='application/json')

    if st.button('Guardar como JSONL'):
        jsonl_data = save_as_jsonl(structured_data, content_key, label_keys, 'output')
        st.download_button(label='Descargar JSONL', data=jsonl_data.encode('utf-8'), file_name=f'{file_name}.jsonl', mime='application/json')

    if st.button('Guardar como CSV'):
        df = save_as_csv(structured_data, content_key, label_keys, 'output')
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label='Descargar CSV', data=csv_data, file_name=f'{file_name}.csv', mime='text/csv')

    if st.button('Guardar como XML'):
        xml_data = save_as_xml(structured_data, content_key, label_keys)
        st.download_button(label='Descargar XML', data=xml_data, file_name=f'{file_name}.xml', mime='application/xml')
