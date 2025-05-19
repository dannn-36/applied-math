import numpy as np
import cv2


def punto1():
    imagen = [
        [241, 237, 240, 242, 237, 248, 254, 251, 250, 251, 250, 250, 238, 241, 241, 240, 241, 234, 254, 253, 249, 251, 253, 253, 251, 236, 241, 240, 239, 242],
        [240, 241, 239, 243, 253, 144, 93, 99, 98, 95, 97, 132, 249, 245, 246, 244, 248, 253, 131, 95, 97, 95, 99, 97, 143, 253, 244, 239, 239, 238],
        [238, 241, 240, 236, 178, 37, 1, 1, 1, 3, 3, 31, 202, 194, 198, 198, 197, 202, 34, 3, 4, 1, 0, 0, 35, 171, 232, 243, 240, 240],
        [242, 241, 243, 222, 3, 1, 1, 2, 4, 2, 6, 28, 95, 84, 84, 86, 85, 90, 31, 6, 2, 4, 0, 1, 1, 5, 220, 242, 241, 238],
        [241, 237, 244, 224, 17, 1, 2, 1, 1, 3, 44, 98, 93, 97, 94, 96, 96, 93, 95, 44, 4, 2, 3, 1, 1, 15, 221, 244, 239, 244],
        [241, 236, 248, 218, 12, 2, 0, 1, 4, 48, 89, 105, 95, 96, 89, 86, 93, 99, 103, 88, 50, 4, 1, 1, 2, 11, 223, 244, 242, 237],
        [239, 244, 239, 228, 14, 1, 2, 5, 39, 94, 102, 89, 93, 93, 128, 128, 92, 95, 90, 104, 92, 43, 3, 1, 1, 17, 220, 247, 242, 239],
        [240, 238, 247, 220, 12, 2, 7, 30, 102, 101, 88, 100, 93, 87, 177, 175, 88, 91, 98, 91, 99, 101, 27, 8, 2, 11, 225, 243, 241, 240],
        [238, 242, 245, 220, 13, 2, 37, 104, 95, 95, 94, 92, 95, 85, 170, 172, 89, 94, 90, 94, 98, 93, 104, 37, 3, 11, 223, 245, 240, 240],
        [238, 240, 240, 225, 1, 135, 168, 89, 95, 84, 159, 229, 235, 105, 165, 162, 103, 236, 232, 162, 83, 98, 88, 168, 140, 2, 221, 245, 240, 242],
        [239, 244, 241, 233, 128, 225, 194, 81, 94, 87, 188, 240, 125, 90, 168, 173, 93, 123, 233, 189, 82, 98, 81, 194, 222, 127, 235, 236, 239, 239],
        [239, 236, 243, 240, 250, 248, 182, 83, 93, 82, 202, 191, 7, 72, 173, 170, 71, 7, 196, 201, 83, 89, 84, 179, 250, 250, 243, 243, 241, 239],
        [242, 236, 241, 239, 243, 244, 179, 82, 93, 87, 189, 204, 12, 84, 174, 175, 82, 9, 201, 194, 83, 97, 88, 177, 246, 241, 234, 238, 236, 242],
        [237, 242, 240, 241, 238, 246, 175, 87, 89, 168, 204, 198, 177, 200, 207, 212, 199, 184, 195, 206, 167, 85, 81, 180, 245, 239, 243, 242, 239, 239],
        [240, 240, 240, 236, 243, 244, 173, 84, 163, 190, 205, 203, 198, 92, 76, 70, 86, 204, 204, 202, 197, 160, 83, 172, 246, 239, 236, 246, 240, 236],
        [239, 240, 241, 242, 238, 244, 201, 148, 207, 203, 198, 202, 196, 7, 5, 5, 9, 189, 202, 202, 197, 208, 152, 197, 244, 240, 243, 235, 239, 242],
        [239, 241, 241, 240, 240, 239, 242, 240, 199, 198, 203, 200, 195, 43, 7, 12, 43, 196, 199, 199, 201, 197, 239, 244, 240, 240, 240, 240, 240, 240],
        [242, 237, 238, 241, 239, 242, 243, 235, 199, 198, 196, 197, 201, 209, 52, 51, 207, 199, 201, 200, 194, 200, 234, 243, 240, 240, 240, 240, 240, 240],
        [239, 240, 241, 240, 237, 233, 248, 248, 230, 203, 195, 200, 202, 220, 51, 49, 217, 203, 200, 195, 205, 227, 239, 239, 240, 240, 240, 240, 240, 240],
        [238, 240, 238, 239, 245, 246, 188, 139, 247, 225, 223, 208, 160, 164, 109, 109, 167, 159, 206, 219, 222, 242, 242, 241, 240, 240, 239, 240, 240, 240],
        [239, 239, 241, 239, 237, 253, 120, 3, 196, 241, 231, 187, 100, 98, 161, 166, 102, 93, 189, 233, 243, 239, 241, 239, 240, 240, 239, 239, 240, 240],
        [241, 236, 240, 239, 239, 250, 142, 3, 4, 204, 174, 80, 91, 97, 96, 92, 90, 97, 83, 159, 244, 242, 240, 238, 240, 240, 239, 240, 240, 240],
        [238, 238, 243, 241, 242, 239, 234, 165, 5, 90, 117, 99, 98, 100, 35, 38, 104, 97, 96, 105, 142, 240, 238, 241, 240, 240, 240, 240, 239, 240],
        [242, 240, 239, 239, 238, 243, 244, 232, 142, 99, 59, 51, 94, 103, 26, 25, 103, 97, 49, 54, 123, 238, 242, 239, 240, 240, 240, 240, 239, 239],
        [240, 240, 240, 240, 240, 240, 241, 240, 247, 136, 20, 9, 101, 100, 27, 23, 100, 102, 7, 23, 133, 237, 240, 240, 240, 240, 240, 240, 240, 240],
        [240, 240, 240, 240, 240, 240, 241, 240, 244, 136, 46, 25, 100, 103, 34, 37, 102, 101, 28, 45, 132, 242, 239, 239, 240, 240, 240, 240, 240, 239],
        [240, 240, 240, 240, 240, 240, 240, 241, 236, 234, 229, 184, 82, 77, 187, 184, 77, 75, 186, 230, 232, 242, 240, 239, 240, 240, 240, 240, 239, 239],
        [240, 240, 240, 240, 240, 240, 240, 240, 241, 243, 245, 234, 195, 185, 228, 232, 189, 196, 237, 246, 243, 236, 241, 240, 240, 240, 240, 240, 239, 239],
        [240, 240, 240, 239, 240, 240, 240, 240, 242, 240, 242, 243, 246, 245, 245, 242, 247, 249, 235, 246, 238, 240, 241, 240, 240, 240, 240, 240, 240, 240],
        [240, 240, 239, 239, 240, 240, 240, 240, 241, 233, 241, 240, 238, 241, 239, 237, 241, 241, 244, 238, 237, 243, 238, 239, 240, 240, 239, 239, 240, 240],
    ]
    # la matriz IMG representa una imagen en escala de grises, donde cada número es un valor de píxel.

    # Se obtiene el número de filas (alto) y columnas (ancho) de la imagen
    filas = len(imagen)
    columnas = len(imagen[0])

    # Se crea una nueva matriz vacía con las dimensiones rotadas (col, fila)
    # Esta será la imagen resultante después de rotar 90 grados
    imagen_rotada = []

    # Iteramos por cada columna de la imagen original
    for j in range(columnas):
        nueva_fila = []

        # Recorremos las filas en orden inverso para rotar la imagen 90° a la derecha
        for i in range(filas - 1, -1, -1):
            nueva_fila.append(imagen[i][j])  # Se toma el valor correspondiente de la fila i y columna j

        imagen_rotada.append(nueva_fila)  # Se añade la nueva fila a la imagen rotada

    # Mostramos la imagen rotada en consola
    for fila in imagen_rotada:
        print(fila)

def punto2():

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

def punto3():


    datos = np.array([
    [ 1. , 25. , 2.5, 4. , 3. , 68.2],
    [ 2. , 32. , 3.2, 3. , 4. , 72.5],
    [ 3. , 28. , 2.0, 5. , 2. , 65.9],
    [ 4. , 45. , 5.0, 2. , 5. , 81.4],
    [ 5. , 38. , 4.2, 3. , 4. , 75.3],
    [ 6. , 22. , 1.8, 6. , 2. , 64.7],
    [ 7. , 31. , 3.5, 4. , 4. , 73.1],
    [ 8. , 40. , 4.8, 2. , 3. , 77.8],
    [ 9. , 27. , 2.2, 5. , 3. , 66.3],
    [10. , 35. , 4.0, 3. , 4. , 74.5],
    [11. , 48. , 5.5, 1. , 5. , 83.6],
    [12. , 29. , 2.8, 4. , 3. , 70.2],
    [13. , 26. , 2.1, 5. , 2. , 65.5],
    [14. , 42. , 4.6, 2. , 4. , 79.1],
    [15. , 37. , 4.1, 3. , 4. , 74.8],
    [16. , 24. , 2.3, 6. , 3. , 67.3],
    [17. , 50. , 5.7, 1. , 5. , 84.5],
    [18. , 33. , 3.6, 4. , 3. , 72.8],
    [19. , 39. , 4.3, 3. , 4. , 76.2],
    [20. , 30. , 3.0, 5. , 3. , 69.9],
    [21. , 41. , 4.7, 2. , 4. , 78.3],
    [22. , 23. , 2.0, 6. , 2. , 65.7],
    [23. , 36. , 4.0, 3. , 4. , 74.2],
    [24. , 34. , 3.9, 3. , 4. , 73.7],
    [25. , 29. , 2.5, 5. , 3. , 68.4],
    [26. , 46. , 5.2, 2. , 5. , 82.1],
    [27. , 21. , 1.7, 6. , 2. , 64.3],
    [28. , 43. , 4.9, 2. , 4. , 80.0],
    [29. , 30. , 2.9, 5. , 3. , 69.1],
    [30. , 38. , 4.2, 3. , 4. , 75.0]
    ])

    X = datos[:, 1:5]  # Características (edad, ingresos, horas de ejercicio, nivel educativo)
    Y = datos[:, [5]]  # Etiqueta (índice de bienestar)

    X = np.hstack([np.ones((X.shape[0], 1)), X]) # Se añade una columna de unos para el intercepto

    #minimos cuadrados
    Xt = X.T # Transpuesta de X
    XtX = Xt @ X # Producto punto entre Xt y X
    XtY = Xt @ Y # Producto punto entre Xt y Y
    beta = np.linalg.inv(XtX) @ XtY # Inversa de XtX multiplicada por XtY
    # beta es un vector que contiene los coeficientes del modelo de regresión lineal.

    print("Coeficientes del modelo (β):")
    print(f"Intercepto:                  {beta[0][0]:.4f}")
    print(f"Edad:                        {beta[1][0]:.4f}")
    print(f"Ingresos Mensuales:          {beta[2][0]:.4f}")
    print(f"Horas Ejercicio x Semana:    {beta[3][0]:.4f}")
    print(f"Nivel Educativo:             {beta[4][0]:.4f}") 

    nueva_persona = np.array([[1, 52, 1.7, 6, 5]]) # Se añade el 1 para el intercepto
    prediccion = nueva_persona @ beta # Producto punto entre la nueva persona y los coeficientes

    print(f"\nPredicción del Índice de Bienestar: {prediccion[0][0]:.2f} (escala 0-100)") 


def punto4():
    # Cada muestra es un vector con 4 features + 1 label (último valor)
    datos = np.array([
        [20, 6, 0, 2, 0.12],
        [30, 4, 1, 3, 0.60],
        [25, 5, 1, 3, 0.80],
        [40, 2, 1, 5, 0.95],
        [22, 7, 0, 5, 0.05],
    ])

    X = datos[:, :4]   # Features
    y = datos[:, 4]    # Labels


    X = X / np.max(X, axis=0) 
    #Se divide cada columna por su valor máximo para que todas las características queden en el rango [0,1].
    #Esto es crucial para que el entrenamiento sea más estable y rápido.



    # Inicializar pesos
    W = np.zeros(4) # Pesos inicializados a 0
    b = 0.0 # Bias inicializado a 0
    lr = 0.1  # Learning rate es 0,1 que es un valor común para la tasa de aprendizaje en algoritmos de optimización.

    # Función sigmoide
    def sigmoid(z):
        return 1 / (1 + np.exp(-z)) # Función sigmoide para la activación

    # Entrenamiento
    for epoch in range(1000):
        for i in range(len(X)):
            z = np.dot(W, X[i]) + b # Producto punto entre los pesos y la muestra + bias
            pred = sigmoid(z) # Predicción
            error = y[i] - pred # Error entre la predicción y el valor real
            
            # Gradiente y actualización
            W += lr * error * X[i] # Actualización de pesos
            b += lr * error # Actualización de bias

    # Resultado final


    print("Pesos entrenados:", W)
    print("Bias entrenado:", b)


    # Predicción

    muestra_nueva = np.array([28, 5, 1, 3])  
    muestra_nueva = muestra_nueva / np.max(X, axis=0) # Nueva muestra normalizada

    muestra_nueva @ W + b # Producto punto entre la nueva muestra y los pesos + bias

    print(sigmoid(muestra_nueva @ W + b))


#menu
def menu():
    while True:
        print("Seleccione una opción:")
        print("1. Rotar imagen 90°")
        print("2. Insertar letra en imagen")
        print("3. Regresión lineal")
        print("4. Perceptrón")
        print("5. Salir")

        opcion = input("Opción: ")

        if opcion == '1':
            punto1()
        elif opcion == '2':
            punto2()
        elif opcion == '3':
            punto3()
        elif opcion == '4':
            punto4()
        elif opcion == '5':
            break
        else:
            print("Opción no válida, intente de nuevo.")
# Ejecutar el menú
menu()
# Fin del programa