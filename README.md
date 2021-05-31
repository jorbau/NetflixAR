# Proyecto Psiv

## Información Importante para ejecutar
- Para ejecutar los scripts hace falta python 3.7 con Opencv <= 3.4.2.16, ya que usa SIFT y este no está disponible en versiones superiores.
### Preparación antes de ejecutar
- Descomprimir carpeta images
-Ejecutar generateFeatures.py <path al directorio de las imagenes descomprimidas>
- Es necesario tener la carpeta features que se generó en el paso anterior y el archivo Netflix_data.json en el mismo directorio que Netflix.py
- Para ejecutar $python Netflix.py <path_imagen> 
  >ejemplo: python Netflix.py foto.jpg
 - El dataset contiene solo imagenes de la categoria peliculas de anime de netflix, por lo que solo funciona en esa categoria, aunque es perfectamente ampliable.

## Información adicional
- El script get_data.py sirve para actualizar el archivo Netflix_data.json, pero es inutil si no se añaden las imagenes de las nuevas películas.
- En la carpeta Resultados hay una excel con resultados obtenidos e imagenes de muestra de los resultados.

 ## Grupo

- Javier Ortega
- Roberto Manresa
 
