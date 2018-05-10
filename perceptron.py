# -*- coding: utf-8 -*-
"""
Spyder Editor

Criado por Hudson Fernando
"""
import numpy as np

#entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) OPERADOR AMD
#saidaEsperada   = np.array([0, 0, 0, 1])

#entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) OPERADOR OR
#saidaEsperada   = np.array([0, 1, 1, 1])

entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) #OPERADOR XOR
saidaEsperada   = np.array([0, 1, 1, 0])
pesos = np.array([0.0, 0.0])
taxaDeAprendizagem = 0.1

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)
# ENCONTRA O MELHOR CONJUNTO DE PESOS PARA DEIXAR A REDE NEURAL100% CORRETA
def treinar():
    erroTotal = 1
    while(erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidaEsperada)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = saidaEsperada[i] - saidaCalculada
            erroTotal += erro
            for j  in range(len(pesos)):
                pesos[j] = pesos[j] +(taxaDeAprendizagem * entradas[i][j]* erro)
                print('Peso atualizado: '+ str(pesos[j]))
        print('Total de erros: '+ str(erroTotal))

treinar()
print('Rede neural treinada')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
