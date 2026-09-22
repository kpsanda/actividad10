import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def entrenar_y_guardar_modelo():
    # 1. Datos de entrenamiento simulados: [frecuencia_compra, monto_promedio, dias_ultima_compra]
    # Clasificación de Riesgo de cliente / Abandono (0: Riesgo Bajo, 1: Riesgo Alto)
    X = np.array(
        [
            [2, 15000, 45],  # cliente inactivo / Riesgo Alto
            [20, 200000, 1],  # Cliente VIP
            [1, 10000, 60],  # cliente en riesgo
            [15, 150000, 2],  # cliente activo y recurrente
            [12, 120000, 5],  # cliente activo
            [13, 25000, 30],  # cliente en riesgo
        ]
    )

    # Etiquetas (0: Bajo Riesgo, 1: Alto Riesgo)
    y = np.array([1, 0, 1, 0, 0, 1])

    # 2. Entrenar el clasificador Random Forest
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # 3. Guardar el modelo en un archivo binario joblib
    ruta_guardado = os.path.join(
        os.path.dirname(__file__), "modelo_clasificacion.joblib"
    )
    joblib.dump(clf, ruta_guardado)
    print(f"Modelo de IA entrenado y guardado exitosamente en: {ruta_guardado}")


if __name__ == "__main__":
    entrenar_y_guardar_modelo()
