import streamlit as st

# --- Configuración y Título de la Aplicación ---
st.set_page_config(page_title="Detector de Emociones con Bagging", layout="centered")

st.title('Sentiment Analyzer 🤖')
st.markdown("""
Esta aplicación utiliza un modelo de **Bagging para detectar emociones** en el texto que ingreses.
¡Escribe algo y descubre la emoción subyacente!
""")

st.markdown("---")

# --- Sección de Entrada de Texto para Detección de Emociones ---
st.header('Análisis de Emociones')

# Campo de entrada de texto largo para la frase o párrafo
user_text = st.text_area(
    "Ingresa el texto que deseas analizar:",
    "Hoy es un día maravilloso y me siento muy feliz de trabajar en este proyecto.",
    height=150,
    help="Escribe una frase, oración o párrafo para que el modelo detecte la emoción."
)

# Botón para activar el análisis
if st.button('Analizar Emoción', key='analyze_button'):
    if user_text:
        st.subheader('Resultado del Análisis:')

        st.info("Procesando el texto con el modelo Bagging...")
        st.write(f"Texto a analizar: **'{user_text}'**")
        st.write("---")
        st.write("**Nota para el desarrollador:** Tu compañero debe integrar la llamada al modelo de detección de emociones aquí. El resultado de la emoción detectada se mostrará en esta sección.")
        st.write("---")

        # Mensaje de éxito o error (ejemplo)
        # st.success("¡Análisis completado!") # Esto se mostrará cuando se integre el modelo
    else:
        st.warning('Por favor, ingresa algún texto para poder analizar la emoción.')

st.markdown("---")
st.caption("Aplicación creada con Streamlit para el proyecto de detección de emociones.")
