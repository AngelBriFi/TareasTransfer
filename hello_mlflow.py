import os
import mlflow

def main():
    # Nombre del experimento
    mlflow.set_experiment("A01796844: Hello World - MLFlow")

    # Iniciar una ejecución (run) de MLflow
    print("Iniciando registro en MLflow...")
    with mlflow.start_run():
        
        # Registrar parámetros (configuraciones del modelo/proceso)
        mlflow.log_param("tipo_modelo", "Regresion_Lineal")
        mlflow.log_param("epochs", 10)
        mlflow.log_param("learning_rate", 0.05)
        
        # Registrar métricas (resultados cuantificables)
        # Simulación de progreso
        for epoch in range(1, 11):
            loss_simulado = 1.0 / epoch  # La pérdida disminuye
            accuracy_simulado = 0.5 + (epoch * 0.04)  # La precisión aumenta
            
            mlflow.log_metric("loss", loss_simulado, step=epoch)
            mlflow.log_metric("accuracy", accuracy_simulado, step=epoch)

        # Crear y registrar un artefacto (un archivo de salida)
        nombre_archivo = "resultado_resumen.txt"
        with open(nombre_archivo, "w") as f:
            f.write("¡Hola MLflow!\n")
            f.write("El entrenamiento trivial ha finalizado con éxito.\n")
            f.write(f"Precisión final: {accuracy_simulado:.2f}\n")
        
        mlflow.log_artifact(nombre_archivo)
        print(f"Artefacto '{nombre_archivo}' registrado con éxito.")
        
        # Limpiar el archivo local después de subirlo a MLflow
        if os.path.exists(nombre_archivo):
            os.remove(nombre_archivo)

    print("Ejecución finalizada. Todos los datos han sido guardados.")

if __name__ == "__main__":
    main()