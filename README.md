# Reconocimiento de *Ochroma pyramidale* (topa) con imágenes de dron multiespectral

Seis cuadernos de Google Colab, explicados **paso a paso**, para reconocer las copas de *Ochroma pyramidale* (topa) en un ortomosaico tomado con un dron DJI Mavic 3 Multispectral. Cada cuaderno usa un algoritmo distinto:

| Cuaderno | Algoritmo | Con qué aprende | Resultado en el examen (30 %) |
|---|---|---|---|
| [1_RF_paso_a_paso](cuadernos/1_RF_paso_a_paso.ipynb) | Random Forest | 35 rasgos de cada copa: bandas, índices de vegetación y textura | Kappa 0,552 |
| [2_SVM_paso_a_paso](cuadernos/2_SVM_paso_a_paso.ipynb) | Máquina de vectores de soporte | Los mismos rasgos | Kappa 0,654 |
| [3_XGBoost_paso_a_paso](cuadernos/3_XGBoost_paso_a_paso.ipynb) | XGBoost | Los mismos rasgos | Kappa 0,620 |
| [4_YOLO_paso_a_paso](cuadernos/4_YOLO_paso_a_paso.ipynb) | YOLO26l-seg | Teselas de 640 píxeles en color natural | mAP50 0,464 |
| [5_ViT_paso_a_paso](cuadernos/5_ViT_paso_a_paso.ipynb) | Vision Transformer | Recortes de cada copa (224 píxeles) | Kappa 0,951 |
| [6_UNet_paso_a_paso](cuadernos/6_UNet_paso_a_paso.ipynb) | U-Net | Teselas de 640 píxeles en color natural | IoU 0,666 |

Con una misma referencia de 1419 copas revisadas (250 de topa), los seis modelos tuvieron un Kappa de 0,736 a 0,805, sin diferencias significativas entre ellos (prueba de McNemar).

## Qué hace cada cuaderno

Cada paso tiene un **Para qué**, un **Qué verás**, la **Fuente** que lo respalda y una celda corta de código. El orden general es:

1. Abrir el ortomosaico y los polígonos de referencia.
2. Separar las copas (segmentación por cuencas o watershed) o cortar teselas.
3. Dividir los datos en el espacio: 70 % para aprender y 30 % para el examen.
4. Entrenar el modelo y examinarlo con la matriz de confusión.
5. Detectar la topa en todo el ortomosaico y hacer el mapa final (umbral de 0,50).

Los cuadernos guardan las salidas de una corrida real, así que se pueden ver sin correrlos.

## Cómo usarlos

1. Copia `topa_pasos.py` (las funciones de todos los pasos) y los cuadernos a una carpeta `PASO_A_PASO` de tu Google Drive.
2. Abre un cuaderno en Google Colab. Para YOLO, Vision Transformer y U-Net, usa un entorno con GPU.
3. En el **Paso 2**, cambia las rutas por las de tus datos.

## Datos que necesitan (no incluidos)

Por su tamaño, los datos no están en este repositorio:

- **Ortomosaico** GeoTIFF de 8 bandas, uint16, en UTM 18 S: bandas 1 a 3 = rojo, verde y azul de la cámara RGB; 4 = verde; 5 = rojo; 6 = borde rojo; 7 = infrarrojo cercano; 8 = canal alfa.
- **Polígonos de referencia** (shapefile) con el campo `ESPECIE`: 1 = *Ochroma pyramidale*, 2 = otras especies.

## Programas

Python en Google Colab con scikit-learn, XGBoost, Ultralytics, PyTorch, timm, rasterio, geopandas y scikit-image.
