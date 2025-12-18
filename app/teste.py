from sympy.parsing.sympy_parser import null


def _testeFunction(self, casa, terreno, preco):
    return casa

class Casa:
    def __init__(self, casa, terreno, preco):
        self.casa = casa
        self.terreno = terreno
        self.preco = preco
    def calcular_preco(self):
        return self.preco*2


minha_casa = Casa('Branca', 41, 2000)


print(minha_casa.calcular_preco())

