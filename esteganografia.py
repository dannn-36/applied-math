import cv2

# Cargar la imagen
img = cv2.imread("mujer.jpg")

# Mostrar valores iniciales de algunos píxeles
print('Antes:', img[0, 0], img[0, 1], img[0, 2])

# Letra a insertar
letra = "U"
seq = bin(ord(letra))[2:].zfill(8)  # "U" -> '01010101'

# Obtener dimensiones
altura, ancho = img.shape[:2]

# Índice de bit actual en la secuencia
bit_index = 0

# Recorrer imagen hasta insertar los 8 bits
for y in range(altura):
    for x in range(ancho):
        for canal in range(3):  # B, G, R en OpenCV
            if bit_index >= len(seq):
                break

            bit = seq[bit_index]
            valor = img[y, x, canal]

            # Si el bit a insertar es '0'
            if bit == '0':
                if valor % 2 != 0:
                    if valor == 255:
                        valor -= 1
                    else:
                        valor += 1
            # Si el bit es '1'
            else:
                if valor % 2 == 0:
                    if valor == 255:
                        valor -= 1
                    else:
                        valor += 1

            # Guardar el valor modificado
            img[y, x, canal] = valor
            bit_index += 1

        if bit_index >= len(seq):
            break
    if bit_index >= len(seq):
        break

# Mostrar valores después de la modificación
print('Después:', img[0, 0], img[0, 1], img[0, 2])

# Mostrar la imagen
cv2.imshow("Imagen con letra U encriptada", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
