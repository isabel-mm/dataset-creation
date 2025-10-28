# 🧠 Conversor de TXT a JSON, JSONL, CSV o XML

## 📘 Descripción general

Esta aplicación permite convertir archivos de texto plano (.txt) en formatos estructurados ampliamente utilizados en análisis lingüístico computacional, minería de texto y procesamiento de lenguaje natural (PLN).

Fue desarrollada con Streamlit, con el objetivo de ofrecer una interfaz sencilla e intuitiva para investigadores, lingüistas y estudiantes que trabajan con corpus y necesitan estructurar sus datos textuales en formatos como JSON, JSONL, CSV o XML.

En el ámbito de la lingüística computacional, la estructuración de datos facilita la aplicación de métodos automáticos y reproducibles para el análisis de corpus, la extracción terminológica, la anotación lingüística y el entrenamiento de modelos supervisados.

## ⚙️ Funcionalidades principales

📂 Carga múltiple de archivos .txt
Permite subir uno o varios textos a la vez.

✂️ Segmentación opcional por oraciones
Utiliza el tokenizador Punkt de NLTK para dividir el texto en unidades oracionales, facilitando el tratamiento posterior en tareas de anotación o modelado.

🧩 Conversión a múltiples formatos estructurados

JSON: lista estructurada de objetos.

JSONL: un objeto JSON por línea, ideal para entrenamiento de modelos.

CSV: formato tabular compatible con Excel y pandas.

XML: estructura jerárquica con etiquetas definidas por el usuario.

🏷️ Definición personalizada de campos y etiquetas
El usuario puede especificar el nombre del campo de contenido (p. ej. "Texto") y las etiquetas o metadatos que desee incluir (p. ej. "autor", "tema", "fecha").

💾 Descarga directa de resultados
Tras el procesamiento, el usuario puede descargar los datos en el formato deseado.

## 🧾 Ejemplo de uso

Sube uno o varios archivos .txt.

(Opcional) Marca la casilla "Segmentar por oraciones" si deseas dividir el texto.

Introduce el nombre del campo de contenido (por defecto: content).

Añade las etiquetas que quieras incluir, separadas por comas (por ejemplo: autor, tema, fecha).

Elige el formato de salida deseado: JSON, JSONL, CSV o XML.

Descarga el archivo generado.
