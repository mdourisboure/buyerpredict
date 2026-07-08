import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

st.set_page_config(
    page_title="PredictBuy Portal",
    page_icon="🛒",
    layout="wide"
)

# ==============================
# CARGAR MODELO
# ==============================

modelo = joblib.load("modelo_random_forest.pkl")

# Usamos las columnas reales con las que fue entrenado el modelo
columnas_modelo = list(modelo.feature_names_in_)

# ==============================
# ENCABEZADO
# ==============================

st.title("🛒 PredictBuy Portal")
st.subheader("Predicción de finalización de compra para ecommerce")

st.write(
    "PredictBuy permite a las tiendas online cargar un dataset de ecommerce "
    "y obtener una predicción sobre qué usuarios tienen mayor probabilidad de finalizar una compra."
)

st.divider()

# ==============================
# CARGA DE DATASET
# ==============================

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

    # ==============================
    # ANÁLISIS AUTOMÁTICO
    # ==============================

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

    # ==============================
    # PREDICCIÓN CON MODELO REAL
    # ==============================

    st.header("3. Predicción de finalización de compra")

    st.info(
        st.info(
    "Las predicciones fueron generadas utilizando un modelo Random Forest entrenado con el 80% del dataset Online Shoppers Purchasing Intention. "
    "Si el archivo cargado contiene la columna Revenue, la app también evalúa el desempeño del modelo sobre datos no vistos durante el entrenamiento."
)
    )

    df_pred = df.copy()

    # Si el CSV contiene la columna Revenue, la eliminamos porque es la variable objetivo
    if "Revenue" in df_pred.columns:
        df_pred = df_pred.drop(columns=["Revenue"])

    # Aplicar las mismas transformaciones usadas en Colab
    columnas_dummy = ["Month", "VisitorType"]
    columnas_dummy_existentes = [col for col in columnas_dummy if col in df_pred.columns]

    df_pred = pd.get_dummies(
        df_pred,
        columns=columnas_dummy_existentes,
        drop_first=True
    )

    # Convertir Weekend a 0/1 si existe
    if "Weekend" in df_pred.columns:
        df_pred["Weekend"] = df_pred["Weekend"].astype(int)

    # Alinear columnas con las columnas usadas al entrenar el modelo
    df_pred = df_pred.reindex(columns=columnas_modelo, fill_value=0)

    # Generar predicciones reales
    probabilidades = modelo.predict_proba(df_pred)[:, 1]
    predicciones = modelo.predict(df_pred)

    # Crear dataframe de resultados
    df_resultado = df.copy()
    df_resultado["probabilidad_compra"] = probabilidades
    df_resultado["prediccion_compra"] = predicciones

    # ==============================
    # EVALUACIÓN SI EXISTE REVENUE
    # ==============================

    if "Revenue" in df.columns:
        st.divider()
        st.header("Evaluación del modelo con datos reales")

        y_real = df["Revenue"].astype(int)
        y_pred_app = df_resultado["prediccion_compra"].astype(int)

        accuracy = accuracy_score(y_real, y_pred_app)
        precision = precision_score(y_real, y_pred_app, zero_division=0)
        recall = recall_score(y_real, y_pred_app, zero_division=0)
        f1 = f1_score(y_real, y_pred_app, zero_division=0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", round(accuracy, 4))

        with col2:
            st.metric("Precision", round(precision, 4))

        with col3:
            st.metric("Recall", round(recall, 4))

        with col4:
            st.metric("F1-score", round(f1, 4))

        matriz = confusion_matrix(y_real, y_pred_app)

        matriz_df = pd.DataFrame(
            matriz,
            index=["Real: No compra", "Real: Compra"],
            columns=["Predicho: No compra", "Predicho: Compra"]
        )

        st.subheader("Matriz de confusión")
        st.dataframe(matriz_df)
    # ==============================
    # CLASIFICACIÓN Y RECOMENDACIONES
    # ==============================

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
        df_resultado[
            [
                "probabilidad_compra",
                "prediccion_compra",
                "segmento",
                "accion_recomendada"
            ]
        ].head(20)
    )

    st.divider()

    # ==============================
    # SEGMENTACIÓN
    # ==============================

    st.header("4. Segmentación de usuarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Alta probabilidad",
            int((df_resultado["segmento"] == "Alta probabilidad").sum())
        )

    with col2:
        st.metric(
            "Probabilidad media",
            int((df_resultado["segmento"] == "Probabilidad media").sum())
        )

    with col3:
        st.metric(
            "Baja probabilidad",
            int((df_resultado["segmento"] == "Baja probabilidad").sum())
        )

    st.bar_chart(df_resultado["segmento"].value_counts())

    st.divider()

    # ==============================
    # RECOMENDACIONES COMERCIALES
    # ==============================

    st.header("5. Recomendaciones comerciales")

    st.markdown(
        """
        - **Alta probabilidad de compra:** simplificar checkout y evitar descuentos innecesarios.
        - **Probabilidad media:** ofrecer incentivo, envío gratis o recordatorio.
        - **Baja probabilidad:** incluir al usuario en campañas de remarketing.
        """
    )

    st.divider()

    # ==============================
    # DESCARGA DE RESULTADOS
    # ==============================

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

