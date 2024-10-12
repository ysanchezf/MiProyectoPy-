# Ejercicio 1: Multiplicar todos los elementos por 3
def multiplica3(lista):
    return list(map(lambda x: x * 3, lista))

numeros = [1, 2, 3, 4, 5]
E1 = multiplica3(numeros)
print("Ejercicio 1:", E1)

# Ejercicio 2: Filtrar números mayores a 10
def FM10(lista):
    return list(filter(lambda x: x > 10, lista))

numeros = [5, 10, 15, 20, 25]
EJ2 = FM10(numeros)
print("Ejercicio 2:", EJ2)

# Ejercicio 3: Convertir una lista de palabras a mayúsculas
def CM(lista):
    return list(map(lambda x: x.upper(), lista))

palabras = ["hola", "mundo", "python"]
EJ3 = CM(palabras)
print("Ejercicio 3:", EJ3)

# Ejercicio 4: Filtrar palabras que empiezan con 'p'
def FCP(lista):
    return list(filter(lambda x: x.startswith('p'), lista))

palabras = ["hola", "mundo", "python", "programacion"]
EJ4 = FCP(palabras)
print("Ejercicio 4:", EJ4)

# Ejercicio 5: Calcular la longitud de cada palabra
def CL(lista):
    return list(map(lambda x: len(x), lista))

palabras = ["hola", "mundo", "python"]
EJ5 = CL(palabras)
print("Ejercicio 5:", EJ5)

# Ejercicio 6: Filtrar palabras con longitud mayor a 4
def FM4(lista):
    return list(filter(lambda x: len(x) > 4, lista))

palabras = ["hola", "mundo", "python", "hi"]
EJ6 = FM4(palabras)
print("Ejercicio 6:", EJ6)

# Ejercicio 7: Convertir una lista de temperaturas de Celsius a Fahrenheit
def CTOF(lista):
    return list(map(lambda x: (x * 9/5) + 32, lista))

celsius = [0, 20, 37, 100]
EJ7 = CTOF(celsius)
print("Ejercicio 7:", EJ7)

# Ejercicio 8: Filtrar temperaturas mayores a 50 grados Fahrenheit
def FM50(lista):
    return list(filter(lambda x: x > 50, lista))

fahrenheit = [32.0, 68.0, 98.6, 212.0]
EJ8 = FM50(fahrenheit)
print("Ejercicio 8:", EJ8)

# Ejercicio 9: Sumar 5 a todos los elementos de una lista
def sumar5(lista):
    return list(map(lambda x: x + 5, lista))

numeros = [1, 2, 3, 4, 5]
EJ9 = sumar5(numeros)
print("Ejercicio 9:", EJ9)

# Ejercicio 10: Filtrar palabras que contienen la letra 'a'
def FCA(lista):
    return list(filter(lambda x: 'a' in x, lista))

palabras = ["hola", "mundo", "python", "vida"]
EJ10 = FCA(palabras)
print("Ejercicio 10:", EJ10)
