# Revisión final contra la rúbrica

**Fecha:** 7 de noviembre de 2026

**Línea base técnica validada:** `a5955e4dea5e6bed9e9583cbbeeb03acece4875d`

**Commit documental empaquetado:** `cb28172e0f74fc93cd8def93a03edda1b9f32114`

## Resultado general

La implementación, medición, validación y preparación documental están completas. Se generaron paquetes independientes para código y resultados, el informe final de 18 páginas, la presentación ejecutiva y un guion extendido de defensa. Los cambios documentales del 7 de noviembre permanecen locales para revisión; después de aprobarlos deben regenerarse los paquetes y realizarse la entrega en Canvas.

## Trazabilidad de la versión

- `a5955e4` es el último commit que modificó código o pruebas.
- `cb28172` es el commit documental usado para generar la primera entrega final verificada.
- Entre ambos commits no cambiaron `src/`, `common/`, `tests/`, `scripts/`, `CMakeLists.txt`, `Makefile` ni los datos oficiales de medición.
- La actualización descrita en este documento no se ha confirmado ni subido a Git; debe revisarse antes de crear un commit.

## Informe - 20 puntos

| Criterio | Evidencia | Estado |
|---|---|---|
| Algoritmo, CUDA y memorias | `docs/informe_final.tex`, secciones 2 a 6 | Completo |
| Formato, introducción, cuerpo, conclusiones y apéndices técnicos | `docs/informe_final.pdf` (18 páginas) | Completo |
| Al menos tres referencias | Duda y Hart; CUDA Programming Guide; CUDA Runtime API; Compute Sanitizer | Completo |
| Diez tiempos por variante | `results/timings.csv` y `BITACORA.md` | Completo |
| Memoria constante y diagrama | Sección 5 y `docs/figures/memoria_constante.pdf` | Completo |
| Memoria compartida y diagrama | Sección 6 y `docs/figures/memoria_compartida.pdf` | Completo |
| Conclusiones y recomendaciones | Sección 11 | Completo |

## Programa y presentación - 80 puntos

| Criterio | Evidencia | Estado |
|---|---|---|
| Global, Constante y Compartida | `src/hough.cu` | Completo |
| Cálculo de ID y coordenadas | Kernels y explicación del informe | Completo |
| Medición con CUDA Events | `run_kernel_mode` y bitácora | Completo |
| Imagen con líneas | `results/runway_lines.png` | Completo |
| Comentarios consistentes | Código, README y scripts | Completo |
| Barreras de bloque | Dos `__syncthreads()` en Compartida | Completo |
| Liberación y reinicio | RAII y `cudaDeviceReset` | Completo |
| Presentación ejecutiva | `docs/presentacion.html` | Completo; ensayar |

## Paquetes preparados

| Archivo | Contenido | Estado |
|---|---|---|
| `01_Codigo_CUDA_Grupo_1.zip` | Código CUDA/C++, construcción, pruebas, scripts y documentación reproducible | Verificado |
| `02_Mediciones_y_Resultados_Grupo_1.zip` | Treinta tiempos, resumen, metadatos, validaciones, Sanitizer e imagen final | Verificado |
| `03_Informe_Final_Proyecto_3_CUDA_Grupo_1.pdf` | Informe final de 18 páginas | Verificado |

## Evidencia adicional

- Referencia CPU y 9,000 bins idénticos en las tres variantes.
- 12,456,540 votos en la imagen oficial.
- CTest: 7/7 pruebas aprobadas.
- Memcheck: 0 errores.
- Racecheck: 0 peligros, 0 errores y 0 advertencias.
- Treinta mediciones, dos calentamientos por variante.
- Seis líneas finales sobre la imagen.

## Pendientes humanos antes de entregar

1. Revisar y aprobar conjuntamente los cambios documentales locales.
2. Ensayar el reparto de `docs/GUIA_EXPOSICION.md` y del guion extendido.
3. Probar `docs/presentacion.html` en la computadora y resolución de la exposición.
4. Regenerar los dos ZIP y el PDF final si se aprueban estas modificaciones.
5. Comprobar los hashes de los archivos regenerados.
6. Subir a Canvas el código, resultados, informe PDF y enlace de GitHub.

## Conclusión

El proyecto cubre los requisitos técnicos y documentales del enunciado. No se requieren cambios adicionales en los kernels CUDA. Solo quedan la aprobación humana de esta revisión, el ensayo, la regeneración de paquetes y la carga final.
