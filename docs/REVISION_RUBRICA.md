# Revisión del avance

**Fecha de revisión:** 30 de octubre de 2026  
**Objetivo:** revisar qué llevamos y qué nos falta antes de la entrega final.

## Cómo vamos

La parte fuerte del proyecto ya está hecha: tenemos las tres versiones CUDA, las mediciones, la comparación con CPU y la imagen con las líneas detectadas. Lo que falta es principalmente revisar el informe, agregar los datos académicos, preparar la presentación y confirmar exactamente qué archivos se subirán a Canvas.

## Informe - 20 puntos

| Criterio                                                                 | Cómo vamos                       | Dónde está                                        | Qué falta                                                    |
| ------------------------------------------------------------------------ | -------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| Algoritmo, implementación CUDA y memorias de GPU                         | Ya está explicado en el borrador | `docs/INFORME_BORRADOR.md`, secciones 2 a 6       | Darle una última revisión a la redacción                     |
| Formato UVG, introducción, cuerpo, conclusiones, apéndice y bibliografía | Casi completo                    | `docs/INFORME_BORRADOR.md`                        | Agregar carnés y revisar que las citas salgan bien en el PDF |
| Diez mediciones para cada variante                                       | Completo                         | `results/timings.csv`, `BITACORA.md`              | Usar la misma desviación estándar en todas las tablas        |
| Explicación y diagrama de memoria constante                              | Incluido en el borrador          | Sección 5 y `docs/figures/memoria_constante.svg`  | Revisar que el diagrama se lea bien al exportar              |
| Explicación y diagrama de memoria compartida                             | Incluido en el borrador          | Sección 6 y `docs/figures/memoria_compartida.svg` | Revisar que el diagrama se lea bien al exportar              |
| Conclusiones y recomendaciones                                           | Incluidas                        | Sección 10                                        | Revisarlas entre los dos antes de cerrar el informe          |

## Programa y presentación - 80 puntos

| Criterio                                          | Cómo vamos        | Dónde está                                               | Qué falta                                                      |
| ------------------------------------------------- | ----------------- | -------------------------------------------------------- | -------------------------------------------------------------- |
| Versiones Global, Constante y Compartida          | Completas         | `src/hough.cu`                                           | Hacer una compilación limpia antes de entregar                 |
| Cálculo de identificadores y coordenadas          | Completo          | Kernels en `src/hough.cu`                                | Nada por ahora                                                 |
| Medición con CUDA Events                          | Completa          | `run_kernel_mode` en `src/hough.cu`                      | Mantener el mismo procedimiento si repetimos las pruebas       |
| Imagen con líneas detectadas                      | Completa          | `results/runway_lines.png`, `results/detected_lines.csv` | Abrirla una vez más y revisar el resultado final               |
| Documentación y comentarios del programa          | Bastante avanzada | Comentarios en `src/hough.cu`, `README.md`               | Revisar que los comentarios sean claros y no repitan demasiado |
| Barreras para los hilos del bloque                | Completo          | Dos llamadas a `__syncthreads()` en el kernel compartido | No cambiar el flujo de los hilos antes de las barreras         |
| Liberación de recursos y reinicio del dispositivo | Completo          | RAII para buffers/eventos y `cudaDeviceReset`            | Nada por ahora                                                 |
| Presentación ejecutiva y exposición oral          | Pendiente         | Todavía no hay una presentación en el repositorio        | Preparar diapositivas, repartir temas y ensayar                |

## Pruebas que ya tenemos

- Treinta mediciones, diez por cada variante.
- Los acumuladores de GPU coinciden exactamente con la referencia de CPU.
- Una prueba de 7 x 5 píxeles que incluye un último bloque incompleto.
- `compute-sanitizer` sin errores de memoria.
- `racecheck` sin peligros, errores ni advertencias.
- El PNG final y el CSV con las líneas seleccionadas.

## Cosas que todavía debemos revisar

1. La guía dice que no basta con compartir el enlace de GitHub; también hay que subir el código.
2. Falta agregar los carnés si los pide el formato de informes de UVG.
3. Hay que volver a exportar el PDF para que incluya los diagramas y las citas del borrador.
4. La desviación estándar debe coincidir en el programa, el resumen, la bitácora y el PDF.
5. El commit que aparece en el informe debería identificar la versión del código que se probó.
6. Todavía falta preparar la presentación.

## Cierre del avance

Con esta revisión damos por cerrado el avance. Se documentó el estado del proyecto frente a la rúbrica, se agregaron las citas y los diagramas al borrador y se aclararon las partes principales de los kernels CUDA. La exportación del PDF definitivo, la presentación y el paquete de entrega quedan para la etapa final del proyecto.