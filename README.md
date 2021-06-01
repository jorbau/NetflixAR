# Proyecto Psiv

## Información Importante para ejecutar
- Para ejecutar los scripts hace falta python 3.7 con Opencv <= 3.4.2.16, ya que usa SIFT y este no está disponible en versiones superiores.
### Preparación antes de ejecutar
- Descomprimir carpeta images
- Ejecutar generateFeatures.py <<path_images>>
  >ejemplo: python generateFeatures.py images
- Es necesario tener la carpeta features que se generó en el paso anterior y el archivo Netflix_data.json en el mismo directorio que Netflix.py
- Para ejecutar $python Netflix.py <<path_imagen>>
  >ejemplo: python Netflix.py Prueba.jpg
  > Se generará una imagen resultado.jpg
 - El dataset contiene solo imagenes de la categoría películas de anime de netflix, por lo que solo funciona en esa categoría, aunque es perfectamente ampliable.

## Información adicional
- El script get_data.py sirve para actualizar el archivo Netflix_data.json, pero es inutil si no se añaden las imagenes de las nuevas películas.
- En la carpeta Resultados hay una excel con resultados obtenidos e imagenes de muestra de los resultados.

 ## Grupo

- Javier Ortega
- Roberto Manresa
 
