# Guía de exposición

## Duración sugerida

Entre 8 y 10 minutos, más preguntas. La presentación HTML contiene 11 diapositivas y se controla con las flechas del teclado, espacio o los botones inferiores.

## Reparto sugerido

### Daniel

- Diapositivas 1 a 5.
- Problema, ecuación de Hough y discretización.
- Diseño CUDA y versión Global.
- Aplicación de memoria Constante.

### Milton

- Diapositivas 6 a 11.
- Memoria Compartida y barreras.
- Protocolo de medición y resultados.
- Validación, imagen final y conclusiones.

Ambos deben conocer todas las secciones porque la evaluación oral es individual.

## Ideas que deben explicarse con claridad

- Un hilo procesa un píxel y calcula su `global_id`.
- El origen se mueve al centro de la imagen y se invierte el eje vertical.
- Cada píxel activo emite 90 votos.
- `atomicAdd` evita pérdida de votos concurrentes.
- La memoria Constante almacena 180 valores `float` de solo lectura y permite broadcast por warp.
- La memoria Compartida crea un acumulador de 36 KB por bloque.
- La primera barrera termina la inicialización y la segunda termina la votación.
- Los hilos fuera de la imagen no pueden retornar antes de las barreras.
- CUDA Events mide el kernel, no las transferencias ni el PNG.
- Compartida fue más rápida en la RTX 4050 probada, pero no garantiza el mismo resultado en cualquier GPU.

## Preguntas probables

### ¿Por qué memoria Constante mejoró tan poco?

Las tablas ocupan solo 720 bytes y pueden beneficiarse de otras cachés. Además, las operaciones atómicas sobre el acumulador global siguen dominando una parte importante del costo.

### ¿Por qué Compartida fue más rápida?

Cada bloque agrupa votos en su acumulador local y reduce la cantidad de actualizaciones directas sobre memoria global. En esta entrada, ese ahorro superó el costo de inicializar, sincronizar y fusionar.

### ¿Para qué sirven las dos barreras?

La primera impide votar antes de terminar la inicialización. La segunda impide fusionar mientras otros hilos todavía modifican el acumulador local.

### ¿Cómo probaron que el resultado es correcto?

Compararon los 9,000 bins de cada variante con una referencia CPU. También ejecutaron casos sintéticos con votos conocidos, memcheck y racecheck.

### ¿Por qué usan operaciones atómicas?

Muchos hilos pueden votar por el mismo bin. Una suma normal podría perder actualizaciones cuando dos hilos escriben simultáneamente.

### ¿Por qué la imagen oficial no prueba el último bloque incompleto?

Porque 800 x 600 = 480,000 es múltiplo exacto de 256. Por eso se agregó la imagen sintética de 7 x 5, cuyo último bloque queda incompleto.

### ¿Qué mide el speedup?

La media de Global dividida entre la media de la variante. Compartida obtuvo `1.082563 / 0.798310 = 1.3561x`.

## Revisión antes de presentar

- Confirmar en pantalla los nombres y carnés: 22193 y 23471.
- Probar la presentación en la computadora que se usará.
- Usar modo de pantalla completa.
- Tener el PDF y la imagen final disponibles como respaldo.
- No leer las diapositivas literalmente.
- Vestimenta business casual, según la rúbrica.
