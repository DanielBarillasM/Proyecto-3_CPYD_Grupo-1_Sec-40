# Revisión final contra la rúbrica

**Fecha:** 6 de noviembre de 2026

**Versión revisada:** commit `a5955e4` más el informe final, la presentación y la guía de exposición.

## Resultado general

La implementación, medición y validación están completas. Se agregaron el informe final, la presentación ejecutiva y la guía de exposición, con nombres y carnés en la portada. Faltan la revisión conjunta, el ensayo y la entrega en Canvas.

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

## Evidencia adicional

- Referencia CPU y 9,000 bins idénticos en las tres variantes.
- 12,456,540 votos en la imagen oficial.
- CTest: 7/7 pruebas aprobadas.
- Memcheck: 0 errores.
- Racecheck: 0 peligros, 0 errores y 0 advertencias.
- Treinta mediciones, dos calentamientos por variante.
- Seis líneas finales sobre la imagen.

## Pendientes antes de entregar

1. Revisar en el PDF final la portada, tablas, figuras, enlaces y numeración.
2. Abrir `docs/presentacion.html` en Chrome o Edge y recorrer todas las diapositivas.
3. Ensayar el reparto de `docs/GUIA_EXPOSICION.md`.
4. Generar el ZIP completo del repositorio desde el último commit.
5. Subir a Canvas el código, informe PDF, presentación y enlace de GitHub.

## Conclusión

El proyecto cubre los requisitos técnicos del enunciado. Solo falta la revisión y entrega final; no se requieren cambios adicionales en los kernels CUDA.
