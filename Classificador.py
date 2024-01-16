from abc import ABC, abstractmethod


class Classificador(ABC):

    @abstractmethod
    def treina(self):
        pass

    @abstractmethod
    def testa(self):
        pass

    @abstractmethod
    def classifica(self, entrada):
        pass
