from Classificador import Classificador
from binarytree import Node
import pandas as pd


class ArvoreDecisao(Classificador):
    def __init__(self, df, target, MIN=20):
        self.MIN = MIN
        self.confusao = None
        self.arvore = None
        self.treino = None
        self.teste = None
        self.porcentagem = None
        self.target = target
        self.df = df

    def treina(self, porcentagem=0.75):
        """treina o modelo"""

        self.porcentagem = porcentagem
        self.treino = self.df.sample(frac=self.porcentagem)
        self.teste = self.df.drop(self.treino.index)
        self.arvore = self._arvore_decisao(self.treino)

    def testa(self):
        """testa o modelo"""

        self.confusao = self._matriz_confusao()

    def _impureza(self, target):
        if not target:
            return 1

        return 1 - (target.count(0) / len(target)) ** 2 - (target.count(1) / len(target)) ** 2

    def _impureza_coluna(self, df, coluna):
        impureza_true = self._impureza(df[coluna == 1][self.target].to_list())
        impureza_false = self._impureza(df[coluna == 0][self.target].to_list())

        media_ponderada = (impureza_true * df[coluna == 1].shape[0] +
                           impureza_false * df[coluna == 0].shape[0]) / df.shape[0]

        return media_ponderada

    def _escolhe_raiz(self, df):
        colunas = list(df.columns)
        colunas.remove(self.target)

        return min(colunas, key=lambda col: self._impureza_coluna(df, df[col]))

    def _construir_folha(self, df):
        return Node(int(df[self.target].mode().iloc[0]))

    def _construir_subarvore(self, df, valor, coluna):
        if df[coluna].nunique() == 1:
            return self._construir_folha(df)

        return self._arvore_decisao(df[df[coluna] == valor].drop(columns=[coluna]))

    def _arvore_decisao(self, df):
        if df.shape[0] < self.MIN:
            return self._construir_folha(df)

        coluna = self._escolhe_raiz(df)
        root = Node(coluna)

        root.right = self._construir_subarvore(df, 0, coluna)
        root.left = self._construir_subarvore(df, 1, coluna)

        return root

    def _classifica(self, arvore, entrada):
        atributo = arvore.value
        if isinstance(atributo, int):
            return atributo

        node = arvore.left if (entrada[atributo] == 1) else arvore.right

        return self._classifica(node, entrada)

    def classifica(self, entrada):
        """classifica uma entrada"""

        return self._classifica(self.arvore, entrada)

    def _matriz_confusao(self):
        confusao = [[0, 0], [0, 0]]

        for _, linha in self.teste.iterrows():
            classificacao = self.classifica(linha)

            verdadeiro_positivo = (classificacao == 1 and linha[self.target] == 1)
            verdadeiro_negativo = (classificacao == 0 and linha[self.target] == 0)
            falso_positivo = (classificacao == 1 and linha[self.target] == 0)
            falso_negativo = (classificacao == 0 and linha[self.target] == 1)

            confusao[0][0] += verdadeiro_positivo
            confusao[0][1] += falso_positivo
            confusao[1][0] += falso_negativo
            confusao[1][1] += verdadeiro_negativo

        return confusao


if __name__ == "__main__":
    df = pd.read_csv("data/agaricus-lepiota.data")
    df = pd.get_dummies(df, drop_first=True)

    ad = ArvoreDecisao(df, target="p_p")
    print("Treinando...")
    ad.treina()
    print(ad.arvore)

    print("Testando...")
    ad.testa()
    print(ad.confusao)
