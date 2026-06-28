
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PredictBuy Portal",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 PredictBuy Portal")
st.subheader("Predicción de finalización de compra para ecommerce")

st.write(
    "PredictBuy permite a las tiendas online cargar un dataset de ecommerce "
    "y obtener una predicción sobre qué usuarios tienen mayor probabilidad de finalizar una compra."
)

st.divider()

st.header("1. Cargar dataset")

archivo = st.file_uploader(
    "Subí un archivo CSV con datos de navegación o comportamiento de usuarios",
    type=["csv"]
)

if archivo is not None:
    df = pd.read_csv(archivo)

    st.success("Dataset cargado correctamente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Filas", df.shape[0])

    with col2:
        st.metric("Columnas", df.shape[1])

    with col3:
        st.metric("Valores nulos", int(df.isnull().sum().sum()))

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    st.divider()

    st.header("2. Análisis automático del dataset")

    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=["object", "bool"]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.write("Variables numéricas detectadas:")
        st.write(columnas_numericas)

    with col2:
        st.write("Variables categóricas detectadas:")
        st.write(columnas_categoricas)

    st.divider()

    st.header("3. Predicción de finalización de compra")

    st.write(
        "En esta versión demo, el portal estima una probabilidad de compra para cada registro. "
        "En una implementación real, esta sección se conectaría con el modelo Random Forest entrenado en Colab."
    )

    np.random.seed(42)
    df_resultado = df.copy()

    df_resultado["probabilidad_compra"] = np.random.uniform(0.05, 0.95, size=len(df_resultado))

    def clasificar(prob):
        if prob >= 0.70:
            return "Alta probabilidad"
        elif prob >= 0.40:
            return "Probabilidad media"
        else:
            return "Baja probabilidad"

    def recomendar(prob):
        if prob >= 0.70:
            return "Facilitar checkout y evitar descuentos innecesarios"
        elif prob >= 0.40:
            return "Ofrecer incentivo: descuento, envío gratis o recordatorio"
        else:
            return "Incluir en campaña de remarketing"

    df_resultado["segmento"] = df_resultado["probabilidad_compra"].apply(clasificar)
    df_resultado["accion_recomendada"] = df_resultado["probabilidad_compra"].apply(recomendar)

    st.subheader("Resultados de predicción")
    st.dataframe(
        df_resultado[["probabilidad_compra", "segmento", "accion_recomendada"]].head(20)
    )

    st.divider()

    st.header("4. Segmentación de usuarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Alta probabilidad", int((df_resultado["segmento"] == "Alta probabilidad").sum()))

    with col2:
        st.metric("Probabilidad media", int((df_resultado["segmento"] == "Probabilidad media").sum()))

    with col3:
        st.metric("Baja probabilidad", int((df_resultado["segmento"] == "Baja probabilidad").sum()))

    st.bar_chart(df_resultado["segmento"].value_counts())

    st.divider()

    st.header("5. Recomendaciones comerciales")

    st.markdown(
        '''
        - **Alta probabilidad de compra:** simplificar checkout y evitar descuentos innecesarios.
        - **Probabilidad media:** ofrecer incentivo, envío gratis o recordatorio.
        - **Baja probabilidad:** incluir al usuario en campañas de remarketing.
        '''
    )

    st.divider()

    st.header("6. Descargar resultados")

    csv = df_resultado.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar predicciones en CSV",
        data=csv,
        file_name="predictbuy_resultados.csv",
        mime="text/csv"
    )

else:
    st.info("Subí un archivo CSV para comenzar.")
