# Resultados generados

Esta carpeta conserva la evidencia de la corrida oficial. El CSV, resumen, información de GPU y PNG principal se versionan junto con el proyecto. Las salidas temporales de Compute Sanitizer se ignoran.

La corrida oficial produce:

- `timings.csv`: diez tiempos por cada una de las tres variantes;
- `summary.md`: resumen estadístico calculado automáticamente desde el CSV;
- `runway_lines.png`: imagen original con máximos de Hough dibujados en rojo;
- `detected_lines.csv`: parámetros y votos de las seis líneas seleccionadas;
- `run_metadata.txt`: fecha, computadora, parámetros y versión de CUDA Toolkit;
- `synthetic_validation.log`: validación GPU de una entrada 7 x 5 con bloque incompleto;
- `gpu_info.txt`: versión del controlador, GPU y estado durante la prueba.
- `memcheck.log`: evidencia de cero errores de memoria.
- `racecheck.log`: evidencia de cero carreras o advertencias.
- `validation.log`: ejecución corta que demuestra igualdad CPU-GPU en los tres modos.

Las mediciones deben ejecutarse con el equipo conectado a corriente, sin otras cargas intensivas de GPU y usando exactamente la misma imagen y configuración para los tres kernels.
