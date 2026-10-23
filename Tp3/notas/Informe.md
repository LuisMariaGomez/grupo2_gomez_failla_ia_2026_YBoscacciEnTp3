# Informe Técnico y Financiero - PrestaMatch (TP3)
## 2. Preparación de Datos:

Teníamos dos fuentes principales de datos:
1. `solicitudes_train.csv`: una fila por cada solicitud con montos, tasas, ingresos, motivo y la variable objetivo `Estado_Final` (1 = Pagado, 0 = Default).
2. `comportamiento_bancario.csv`: el historial mensual de los usuarios (hasta 24 meses por persona con saldos y días de atraso).

### Merge directo y Data Leakage
Si hacíamos un merge directo entre las dos tablas por `ID_Usuario`, cada solicitud se multiplicaba por la cantidad de meses bancarios de la persona (de unas 32.000 solicitudes pasábamos a más de 700.000 filas). Pero estaríamos usando información del futuro, meses posteriores a la fecha en que la persona pidió el préstamo.

Para resolverlo hicimos:
1. **Filtro temporal estricto:** Nos quedamos únicamente con los registros donde `Mes_Referencia < Fecha_Solicitud`. Tambien dejamos afuera el mismo mes de la solicitud para usar solo lo que el banco realmente sabía en ese momento.
2. **Resumen por solicitud:** Agrupamos por `ID_Solicitud` y resumimos el historial previo de cada cliente:
   * Cantidad de meses de historial disponible.
   * Saldo promedio histórico, saldo inicial y saldo final (último conocido).
   * Días de atraso promedio y días de atraso máximo en otras deudas.
3. **Ingeniería de variables (Ratios):**
   * `ratio_prestamo_ingreso`: cuántos meses de sueldo representa el préstamo pedido. Cuanto más alto, más ahogado está el cliente.
   * `ratio_prestamo_saldo`: relación entre el préstamo y el saldo que suele mantener en la cuenta.
   * `ratio_tendencia_saldo`: compara el último saldo con el primero para ver si la persona venía mejorando o quemando ahorros.
4. **Transformación logarítmica:** Como los ratios tenían distribuciones muy asimétricas con valores extremos (outliers), les aplicamos `np.log1p` para normalizarlos y que las variables queden mejor distribuidas antes de entrar al modelo.

---

## 3. División de Datos: Validación sin compartir usuarios

Para evaluar el modelo separamos un 30% de los datos para validación.

No podíamos usar un `train_test_split` común y corriente porque un mismo usuario puede haber pedido más de un préstamo a lo largo del tiempo. Si un usuario aparecía con un préstamo en entrenamiento y con otro en validación, el modelo haría trampa memorizando al cliente en vez de aprender patrones generales.

Por eso usamos **`GroupShuffleSplit`** agrupando por `ID_Usuario`:
* Garantizamos que **0 usuarios sean compartidos** entre entrenamiento y validación.
* Verificamos que la proporción de préstamos pagados se mantenga balanceada en ambos conjuntos (~79% de pago en train y validación).

---

## 4. Red Neuronal (MLP) y Preprocesamiento

Para modelar el riesgo crediticio elegimos una red neuronal **Multi-Layer Perceptron (MLP)** usando `scikit-learn` dentro de un Pipeline.

### A. Preprocesamiento con `ColumnTransformer`
* **`StandardScaler` (Columnas numéricas):** Resta la media y divide por el desvío estándar. Deja todas las variables centradas en 0 con la misma escala para que un monto de $50.000 no opaque a una tasa de interés de 0.15.
* **`OneHotEncoder(handle_unknown='ignore')` (`Motivo_Prestamo`):** Convierte las categorías de texto en columnas binarias (0 y 1) sin asumir un orden arbitrario.

Al meter todo en un `Pipeline`, el escalado se calcula únicamente con `X_train` y se aplica automáticamente a validación, evitando cualquier fuga de datos.

### B. Arquitectura de la Red (`MLPClassifier`)
* **Capas ocultas:** `hidden_layer_sizes=(32, 16)`. Dos capas intermedias que van combinando las variables para extraer relaciones no lineales complejas.
* **Activación intermedia:** `activation='relu'`, estándar para no linealidad.
* **Optimizador:** `solver='adam'`, que ajusta la velocidad de aprendizaje de forma adaptativa.
* **Control de sobreajuste:** `early_stopping=True`, frena el entrenamiento si la red deja de mejorar en validación interna.
* **Capa de salida (Sigmoide):** Al tener una variable binaria (`0` y `1`), Scikit-Learn asigna automáticamente la función **Sigmoide** con pérdida *binary cross-entropy*, produciendo probabilidades continuas entre 0.0 y 1.0.
---

## 5. ¿Por qué no alcanza con `.predict()`?

El método tradicional `.predict()` toma un umbral rígido de 0.5: si la probabilidad supera el 50%, aprueba el crédito. 

En un negocio financiero, **usar un umbral de 0.5 es fundirse seguro**:
* Si un cliente pide $10.000 USD al 15% de tasa y tiene un 70% de chances de pagar, `.predict()` diría *"aprobado"*.
* Pero el 30% de las veces que falla perdemos $10.000 enteros, mientras que el 70% de las veces solo ganamos $1.500. El balance daría pérdida asegurada.

Por eso usamos `.predict_proba para obtener la probabilidad exacta de pago y calculamos la Esperanza Matemática, si la ganancia esperada da negativa, el préstamo se descarta de una, sin importar qué tan simpático parezca el cliente.

### Retorno de Inversión esperado (`roi_esperado`)
Para no desperdiciar el presupuesto de $2.000.000 USD en préstamos grandes que dejan pocos intereses, dividimos la ganancia esperada por el monto solicitado, esto nos dice cuanto esperamos ganar por cada dólar prestado, permitiéndonos ordenar las solicitudes de la más eficiente a la menos eficiente.

---

## 6. Asignación del Presupuesto y Resultados en Validación

Con las solicitudes ordenadas de mayor a menor según su roi_esperado, aplicamos una suma acumulada (`cumsum`) sobre el Monto_Solicitado hasta alcanzar los $2.000.000 USD.

Evaluando contra `Estado_Final` real que ocurrió en validación:
1. **Filtro de calidad:** La tasa de cumplimiento de los clientes seleccionados por la red sube drásticamente en comparación con el promedio general del mercado.
2. **Uso eficiente del dinero:** Logramos colocar prácticamente la totalidad de los $2.000.000 USD sin pasarnos del límite fijado por el directorio.
3. Rentabilidad neta: Al priorizar por ROI esperado y esperanza matemática positiva, los intereses cobrados superan las pérdidas por los pocos defaults que se escapan, logrando un balance neto positivo y un ROI de dos dígitos.



