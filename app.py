import streamlit as st
import joblib # Necesario para cargar el archivo .pkl guardado con joblib
from sklearn.feature_extraction.text import TfidfVectorizer # Importa un ejemplo de vectorizador
import pandas as pd # Para leer el archivo de texto

# --- Configuración y Título de la Aplicación ---
st.set_page_config(page_title="Detector de Emociones con Bagging", layout="centered")

st.title('Sentiment Analyzer 🤖')
st.markdown("""
Esta aplicación utiliza un modelo de **Bagging para detectar emociones** en el texto que ingreses.
¡Escribe algo y descubre la emoción subyacente!
""")

st.markdown("---")

# --- Cargar el Modelo de Detección de Emociones y el Vectorizador ---
emotion_model = None
vectorizer = None

# Cargar el modelo
try:
    with open('final_rf_model.pkl', 'rb') as file:
        emotion_model = joblib.load(file)
    st.success("Modelo de detección de emociones cargado exitosamente. ¡Listo para analizar!")
except FileNotFoundError:
    st.error("Error: El archivo 'final_rf_model.pkl' no se encontró.")
    st.error("Asegúrate de que esté en la misma carpeta que 'app.py'.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.warning("Esto podría indicar que el archivo 'final_rf_model.pkl' está corrupto, vacío, o no es un archivo .pkl válido.")
    st.warning("Asegúrate de que el modelo fue guardado con `joblib` si estás intentando cargarlo con él.")
    st.warning("Pídele a tu compañero que verifique cómo fue guardado el modelo y si el archivo no está dañado.")

# Cargar el vectorizador (¡MUY IMPORTANTE para el preprocesamiento del texto!)
# Se asume que el vectorizador se guardó como 'vectorizer.pkl'
try:
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = joblib.load(file)
    st.success("Vectorizador cargado exitosamente. ¡Listo para preprocesar el texto!")
except FileNotFoundError:
    st.error("Error: El archivo 'vectorizer.pkl' no se encontró.")
    st.error("Es **fundamental** que el vectorizador esté en la misma carpeta que 'app.py'.")
    st.error("Sin el vectorizador, el modelo no puede procesar texto crudo.")
    vectorizer = None
except Exception as e:
    st.error(f"Error al cargar el vectorizador: {e}")
    st.warning("Esto podría indicar que el archivo 'vectorizer.pkl' está corrupto o no es un .pkl válido.")
    st.warning("Pídele a tu compañero el vectorizador exacto utilizado para entrenar el modelo.")
    vectorizer = None

# Advertencia sobre la versión de scikit-learn (basado en tu error)
if st.session_state.get('sklearn_version_warning_shown', False) == False:
    st.warning(
        "Advertencia: Si ves un mensaje sobre la versión de scikit-learn (ej. 'using version 1.7.0'), "
        "podría haber una incompatibilidad entre la versión con la que se entrenó el modelo "
        "y la que tienes instalada. Esto podría llevar a resultados inesperados. "
        "Consulta la documentación de scikit-learn sobre persistencia de modelos para más información."
    )
    st.session_state['sklearn_version_warning_shown'] = True # Para mostrar la advertencia solo una vez

st.markdown("---")

# --- Mapeo de Números a Etiquetas de Emoción ---
# ESTE ES UN EJEMPLO. TU COMPAÑERO DEBE CONFIRMARTE EL MAPEO REAL.
# Si el modelo predice -1 para "negativo", 0 para "neutral" y 1 para "positivo", úsalo así.
# Si tiene otras etiquetas (ej. 'tristeza', 'alegría', 'sorpresa'), las agregas aquí.
emotion_labels = {
    -1: "Negativo 😠",
    0: "Neutral 😐",
    1: "Positivo 😊"
}

# --- Sección de Entrada de Texto para Detección de Emociones ---
st.header('Análisis de Emociones')

# Opción para elegir entre entrada manual o carga de archivo
input_method = st.radio(
    "Elige cómo deseas ingresar el texto:",
    ('Ingresar Texto Manualmente', 'Subir Archivo de Texto (.txt)'),
    index=0 # Por defecto, selecciona "Ingresar Texto Manualmente"
)

user_texts_to_analyze = [] # Lista para almacenar los textos a analizar

if input_method == 'Ingresar Texto Manualmente':
    # Campo de entrada de texto largo para la frase o párrafo
    user_text = st.text_area(
        "Ingresa el texto que deseas analizar:",
        "This product exceeded all my expectations! It's incredibly well-designed and performs flawlessly. Highly recommend!",
        height=150,
        help="Escribe una frase, oración o párrafo para que el modelo detecte la emoción."
    )
    if user_text:
        user_texts_to_analyze.append(user_text)
else: # Subir Archivo de Texto
    uploaded_file = st.file_uploader("Sube un archivo de texto (.txt) con comentarios:", type="txt")
    if uploaded_file is not None:
        # Para leer el archivo línea por línea si cada comentario es una línea
        # O simplemente leer todo el contenido como un solo bloque si es necesario
        string_data = uploaded_file.getvalue().decode("utf-8")
        # Asumiendo que cada línea del archivo es un comentario separado
        user_texts_to_analyze = string_data.splitlines()
        user_texts_to_analyze = [text.strip() for text in user_texts_to_analyze if text.strip()] # Limpiar líneas vacías
        st.info(f"Se han cargado {len(user_texts_to_analyze)} comentarios del archivo.")
        st.text_area("Contenido del archivo (primeras 5 líneas):", "\n".join(user_texts_to_analyze[:5]), height=150, disabled=True)


# Botón para activar el análisis
if st.button('Analizar Emoción', key='analyze_button'):
    if emotion_model is None or vectorizer is None:
        st.error("No se pudo realizar el análisis porque el modelo o el vectorizador no se cargaron correctamente.")
    elif not user_texts_to_analyze:
        st.warning('Por favor, ingresa algún texto o sube un archivo para poder analizar la emoción.')
    else:
        st.subheader('Resultados del Análisis:')
        
        # Iterar sobre cada texto cargado (ya sea uno manual o varios del archivo)
        for i, text_to_analyze in enumerate(user_texts_to_analyze):
            st.write(f"**Comentario {i+1}:** {text_to_analyze}")
            try:
                st.info("Preparando el texto para el análisis (preprocesamiento)...")
                # --- PREPROCESAMIENTO REAL DEL TEXTO USANDO EL VECTORIZADOR CARGADO ---
                # El vectorizador transformará el texto crudo en una representación numérica
                processed_text = vectorizer.transform([text_to_analyze])

                # Realizar la predicción con el modelo
                numerical_emotion = emotion_model.predict(processed_text)[0] # Obtiene el número (-1, 0, 1, etc.)

                # --- Traducir el número a una etiqueta legible ---
                detected_emotion_label = emotion_labels.get(numerical_emotion, f"Emoción Desconocida ({numerical_emotion})")

                # Mostrar el resultado
                if numerical_emotion == -1:
                    st.error(f"La emoción detectada es: **{detected_emotion_label}**")
                elif numerical_emotion == 0:
                    st.info(f"La emoción detectada es: **{detected_emotion_label}**")
                elif numerical_emotion == 1:
                    st.success(f"La emoción detectada es: **{detected_emotion_label}**")
                else: # Para cualquier otro número no mapeado
                    st.warning(f"La emoción detectada es: **{detected_emotion_label}**")

                st.write("---") # Separador para cada análisis
                
            except Exception as e:
                st.error(f"Ocurrió un error durante la predicción para el comentario '{text_to_analyze}': {e}")
                st.warning("Asegúrate de que el texto de entrada se esté procesando correctamente con el vectorizador.")
                st.warning("Verifica que el `vectorizer.pkl` y `final_rf_model.pkl` sean compatibles.")
                st.write("---") # Separador incluso si hay error

st.markdown("---")
st.caption("Aplicación creada con Streamlit para el proyecto de detección de emociones.")
