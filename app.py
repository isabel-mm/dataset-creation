import streamlit as st
import pandas as pd
import json
import os
import nltk
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from nltk.tokenize import PunktSentenceTokenizer
import urllib.request
import zipfile

# --- CONFIGURACIÓN DE NLTK ---
@st.cache_resource
def setup_nltk():
    nltk_data_path = './nltk_data'
    if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
        os.makedirs(nltk_data_path, exist_ok=True)
        url = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip'
        zip_path = os.path.join(nltk_data_path, 'punkt.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(nltk_data_path)
    nltk.data.path.append(nltk_data_path)
    return PunktSentenceTokenizer()

punkt_tokenizer = setup_nltk()

# --- FUNCIONES DE PROCESAMIENTO ---
def process_txt_files(uploaded_files, segment_by_sentences):
    structured_data = []
    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            content = uploaded_file.read().decode('latin-1')
            
        file_name = uploaded_file.name
        if segment_by_sentences:
            sentences = punkt_tokenizer.tokenize(content)
            for sentence in sentences:
                structured_data.append({'fuente': file_name, 'contenido': sentence.strip()})
        else:
            structured_data.append({'fuente': file_name, 'contenido': content.strip()})
    return structured_data

def save_as_xml(data, content_key, label_keys):
    root = ET.Element('corpus')
    for item in data:
        entry = ET.SubElement(root, 'documento')
        content_element = ET.SubElement(entry, content_key)
        content_element.text = item['contenido']
        for key in label_keys:
            label_element = ET.SubElement(entry, key)
            label_element.text = 'PENDIENTE'
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = parseString(xml_str)
    return dom.toprettyxml(indent='  ')

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="PLN Data Structurer", page_icon="🤖")

st.title('Pipeline de Estructuración de Datos para PLN')

st.markdown("""
En el **Procesamiento del Lenguaje Natural (PLN)**, la calidad de los modelos (como Transformers o LLMs) depende directamente de la estructura del *dataset*. Convertir texto plano en formatos estructurados es el primer paso esencial para cualquier tarea de minería de texto, análisis de sentimiento o entrenamiento supervisado.

Esta herramienta transforma archivos `.txt` crudos en formatos interoperables, permitiendo una recuperación de información eficiente y una preparación de datos estandarizada para flujos de trabajo científicos.
""")

with st.expander('📊 Arquitectura de los Formatos'):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write('**JSON / JSONL:** Ideales para modelos de Machine Learning y bases de datos NoSQL. El formato JSONL es el estándar para entrenar modelos con grandes volúmenes de datos línea a línea.')
        st.write('**CSV:** El estándar para análisis estadístico y manipulación con librerías como Pandas o herramientas de hojas de cálculo.')
    with col_b:
        st.write('**XML:** Crucial en proyectos que requieren metadatos jerárquicos complejos o compatibilidad con estándares de anotación lingüística.')

st.divider()

# --- CONFIGURACIÓN DE CARGA ---
uploaded_files = st.file_uploader('Cargar archivos de texto (.txt)', type=['txt'], accept_multiple_files=True)

if uploaded_files:
    st.sidebar.header("Configuración del Dataset")
    segment_by_sentences = st.sidebar.checkbox('Tokenización por oraciones (Punkt)', value=True)
    content_key = st.sidebar.text_input('Etiqueta de contenido (key)', value='texto')
    labels_input = st.sidebar.text_input('Etiquetas de metadatos (separadas por comas)', value='sentimiento, categoria')
    file_output_name = st.sidebar.text_input('Nombre del archivo de salida', value='dataset_procesado')

    label_keys = [label.strip() for label in labels_input.split(',')] if labels_input else []
    
    # Procesamiento
    with st.spinner('Procesando tokens...'):
        structured_data = process_txt_files(uploaded_files, segment_by_sentences)
        df = pd.DataFrame([{content_key: item['contenido'], **{key: '' for key in label_keys}} for item in structured_data])

    st.success(f"Procesamiento completado: {len(structured_data)} registros generados.")

    # Vista previa
    st.write("### Vista previa del Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()
    
    # --- BOTONES DE DESCARGA ---
    st.write("### Exportar Dataset")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        json_str = json.dumps([{content_key: item['contenido'], **{key: [] for key in label_keys}} for item in structured_data], indent=4, ensure_ascii=False)
        st.download_button('Descargar JSON', data=json_str, file_name=f'{file_output_name}.json', mime='application/json', use_container_width=True)

    with c2:
        jsonl_str = '\n'.join([json.dumps({content_key: item['contenido'], **{key: [] for key in label_keys}}, ensure_ascii=False) for item in structured_data])
        st.download_button('Descargar JSONL', data=jsonl_str, file_name=f'{file_output_name}.jsonl', mime='application/jsonl', use_container_width=True)

    with c3:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button('Descargar CSV', data=csv_data, file_name=f'{file_output_name}.csv', mime='text/csv', use_container_width=True)

    with c4:
        xml_data = save_as_xml(structured_data, content_key, label_keys)
        st.download_button('Descargar XML', data=xml_data, file_name=f'{file_output_name}.xml', mime='application/xml', use_container_width=True)
