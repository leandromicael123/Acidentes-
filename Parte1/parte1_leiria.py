# Parte 1 - Acidentes rodoviários no distrito de Leiria: 2019 e 2023 (Grupo 20)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

DISTRITO = "Leiria"
DIAS = ["Segunda-Feira", "Terça-Feira", "Quarta-Feira", "Quinta-Feira", "Sexta-Feira", "Sábado", "Domingo"]


# 1. Leitura dos dados

acid_2019 = pd.read_excel("dados/acidentes-2019.xlsx", sheet_name="30 Dias _ Acidentes")
peoes_2019 = pd.read_excel("dados/acidentes-2019.xlsx", sheet_name="30 Dias_Peões")

acid_2023 = pd.read_excel("dados/acidentes-2023.xlsx", sheet_name="30dias_acV", header=1)
peoes_2023 = pd.read_excel("dados/acidentes-2023.xlsx", sheet_name="30dias_peao", header=1)

acid_2023 = acid_2023.drop(columns=["Unnamed: 0"])
peoes_2023 = peoes_2023.drop(columns=["Unnamed: 0"])
acid_2023.columns = acid_2023.columns.str.replace(" SUM", "")

acid_2019 = acid_2019.dropna(subset=["Id. Acidente"])
acid_2023 = acid_2023.dropna(subset=["Id. Acidente"])
peoes_2019 = peoes_2019.dropna(subset=["Id. Acidente"])
peoes_2023 = peoes_2023.dropna(subset=["Id. Acidente"])

print("Acidentes no país: 2019 =", len(acid_2019), "| 2023 =", len(acid_2023))
print("Peões no país: 2019 =", len(peoes_2019), "| 2023 =", len(peoes_2023))


# 2. Preparação dos dados

def preparar(acid, peoes):
    acid = acid.copy()
    acid["Distrito"] = acid["Distrito"].str.strip()
    acid["Id. Acidente"] = acid["Id. Acidente"].astype("int64")
    peoes = peoes.copy()
    peoes["Id. Acidente"] = peoes["Id. Acidente"].astype("int64")

    data = pd.to_datetime(acid["Datahora"], format="%Y:%m:%d %H:%M:%S")
    acid["month"] = data.dt.month
    acid["hora"] = data.dt.hour
    acid["dia"] = acid["Dia da Semana"].str.strip()
    acid["condicoes"] = acid["Cond Aderência"].str.strip()
    acid["curva"] = acid["Traçado 1"] == "Curva"
    acid["grave"] = (acid["Num. Mortos a 30 dias"] + acid["Num. Feridos graves a 30 dias"]) > 0

    peoes_por_acidente = peoes["Id. Acidente"].value_counts()
    acid["peoes"] = acid["Id. Acidente"].map(peoes_por_acidente).fillna(0).astype(int)
    return acid


acid_2019 = preparar(acid_2019, peoes_2019)
acid_2023 = preparar(acid_2023, peoes_2023)


cont_2019 = acid_2019[~acid_2019["Distrito"].str.startswith("Ilha")]
cont_2023 = acid_2023[~acid_2023["Distrito"].str.startswith("Ilha")]

leiria_2019 = acid_2019[acid_2019["Distrito"] == DISTRITO]
leiria_2023 = acid_2023[acid_2023["Distrito"] == DISTRITO]
print("Acidentes em Leiria: 2019 =", len(leiria_2019), "| 2023 =", len(leiria_2023))


# 3. Tabelas mensais (Quadros 1 e 2)

def mais_frequente(valores, ordem):
    contagens = valores.value_counts()
    empatados = contagens[contagens == contagens.max()].index
    return max(empatados, key=ordem.index)


def tabela_mensal(dados):
    linhas = []
    for mes in range(1, 13):
        m = dados[dados["month"] == mes]
        linhas.append({
            "month": mes,
            "acidentes": len(m),
            "mortos": m["Num. Mortos a 30 dias"].sum(),
            "feridos graves": m["Num. Feridos graves a 30 dias"].sum(),
            "feridos ligeiros": m["Num. Feridos ligeiros a 30 dias"].sum(),
            "pior dia": mais_frequente(m["dia"], DIAS),
            "pior hora": mais_frequente(m["hora"], list(range(24))),
            "condicoes": mais_frequente(m["condicoes"], sorted(m["condicoes"].unique())),
            "acidentes em curva": m["curva"].sum(),
            "acidentes graves": m["grave"].sum(),
            "peoes": m["peoes"].sum(),
            "peoes>0": (m["peoes"] > 0).sum(),
            "curva e peoes>0": (m["curva"] & (m["peoes"] > 0)).sum(),
        })
    return pd.DataFrame(linhas)


tabela_2019 = tabela_mensal(leiria_2019)
tabela_2023 = tabela_mensal(leiria_2023)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\nQuadro 1 - Leiria 2019\n", tabela_2019.to_string(index=False))
print("\nQuadro 2 - Leiria 2023\n", tabela_2023.to_string(index=False))

tabela_2019.to_csv("leiria_2019.csv", index=False, sep=";", encoding="utf-8-sig")
tabela_2023.to_csv("leiria_2023.csv", index=False, sep=";", encoding="utf-8-sig")


# 4. Validação com o exemplo do enunciado (Leiria 2019, meses 1, 2, 3, 11 e 12)

enunciado = pd.DataFrame({
    "month": [1, 2, 3, 11, 12],
    "acidentes": [157, 127, 139, 156, 137],
    "mortos": [5, 6, 1, 2, 6],
    "feridos graves": [12, 8, 10, 9, 12],
    "feridos ligeiros": [180, 154, 164, 188, 161],
    "pior dia": ["Quinta-Feira", "Segunda-Feira", "Sexta-Feira", "Sexta-Feira", "Sábado"],
    "pior hora": [8, 19, 17, 18, 18],
    "condicoes": ["Seco e limpo", "Seco e limpo", "Seco e limpo", "Molhado", "Seco e limpo"],
    "acidentes em curva": [41, 31, 38, 38, 33],
    "acidentes graves": [15, 8, 9, 9, 17],
    "peoes": [24, 14, 17, 23, 14],
    "peoes>0": [24, 14, 16, 22, 14],
    "curva e peoes>0": [3, 0, 1, 2, 2],
})
calculado = tabela_2019[tabela_2019["month"].isin(enunciado["month"])].reset_index(drop=True)
diferencas = (calculado != enunciado).sum().sum()
print("\nCélulas diferentes do exemplo do enunciado:", diferencas)


# 5. Totais anuais e variação (Quadro 3)

def totais(dados):
    return pd.Series({
        "Acidentes com vítimas": len(dados),
        "Mortos": dados["Num. Mortos a 30 dias"].sum(),
        "Feridos graves": dados["Num. Feridos graves a 30 dias"].sum(),
        "Feridos ligeiros": dados["Num. Feridos ligeiros a 30 dias"].sum(),
        "Acidentes graves": dados["grave"].sum(),
        "Acidentes em curva": dados["curva"].sum(),
        "Peões envolvidos": dados["peoes"].sum(),
        "Acidentes com peões": (dados["peoes"] > 0).sum(),
        "Acidentes em curva com peões": (dados["curva"] & (dados["peoes"] > 0)).sum(),
    })


quadro3 = pd.DataFrame({"2019": totais(leiria_2019), "2023": totais(leiria_2023)})
quadro3["Variação (%)"] = (100 * (quadro3["2023"] / quadro3["2019"] - 1)).round(1)
print("\nQuadro 3 - Totais anuais\n", quadro3)

for ano, leiria, cont in [(2019, leiria_2019, cont_2019), (2023, leiria_2023, cont_2023)]:
    print(f"\n{ano}:")
    print("  % acidentes graves - Leiria:", round(100 * leiria["grave"].mean(), 1), "| Continente:", round(100 * cont["grave"].mean(), 1))
    print("  % acidentes em curva - Leiria:", round(100 * leiria["curva"].mean(), 1), "| Continente:", round(100 * cont["curva"].mean(), 1))
    print("  Índice de gravidade (mortos por 100 acidentes):", round(100 * leiria["Num. Mortos a 30 dias"].sum() / len(leiria), 1))
    print("  % acidentes com piso molhado:", round(100 * (leiria["condicoes"] == "Molhado").mean(), 1))
    print("  % acidentes ao fim de semana:", round(100 * leiria["dia"].isin(["Sábado", "Domingo"]).mean(), 1))


# 6. Figuras 1 e 2

meses = np.arange(1, 13)
largura = 0.4

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(meses - largura / 2, tabela_2019["acidentes"], largura, label="2019", color="tab:blue")
ax1.bar(meses + largura / 2, tabela_2023["acidentes"], largura, label="2023", color="tab:red")
ax1.set_xticks(meses)
ax1.set_xlabel("Mês")
ax1.set_ylabel("Acidentes com vítimas")
ax1.legend()
ax2.bar(meses - largura / 2, tabela_2019["acidentes graves"], largura, color="tab:blue")
ax2.bar(meses + largura / 2, tabela_2023["acidentes graves"], largura, color="tab:red")
ax2.set_xticks(meses)
ax2.set_xlabel("Mês")
ax2.set_ylabel("Acidentes graves")
fig.tight_layout()
fig.savefig("figura1_mensal.png", dpi=150)

dia_2019 = 100 * leiria_2019["dia"].value_counts(normalize=True).reindex(DIAS)
dia_2023 = 100 * leiria_2023["dia"].value_counts(normalize=True).reindex(DIAS)
hora_2019 = 100 * leiria_2019["hora"].value_counts(normalize=True).reindex(range(24), fill_value=0)
hora_2023 = 100 * leiria_2023["hora"].value_counts(normalize=True).reindex(range(24), fill_value=0)

x = np.arange(7)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(x - largura / 2, dia_2019, largura, label="2019", color="tab:blue")
ax1.bar(x + largura / 2, dia_2023, largura, label="2023", color="tab:red")
ax1.set_xticks(x)
ax1.set_xticklabels([d[:3] for d in DIAS])
ax1.set_ylabel("% dos acidentes do ano")
ax1.legend()
ax2.plot(range(24), hora_2019, color="tab:blue", label="2019")
ax2.plot(range(24), hora_2023, color="tab:red", label="2023")
ax2.set_xticks(range(0, 24, 3))
ax2.set_xlabel("Hora do dia")
ax2.set_ylabel("% dos acidentes do ano")
fig.tight_layout()
fig.savefig("figura2_dia_hora.png", dpi=150)


# 7. Associação entre o ano e cada distribuição (Quadro 4)
# V de Cramér e valor-p do teste do qui-quadrado 

ambos = pd.concat([leiria_2019.assign(ano=2019), leiria_2023.assign(ano=2023)], ignore_index=True)
ambos["periodo"] = pd.cut(ambos["hora"], bins=[-1, 3, 7, 11, 15, 19, 23],
                          labels=["0-3 h", "4-7 h", "8-11 h", "12-15 h", "16-19 h", "20-23 h"])
ambos["com peoes"] = ambos["peoes"] > 0

variaveis = {
    "dia da semana": "dia",
    "período do dia": "periodo",
    "condições de aderência": "condicoes",
    "em curva": "curva",
    "acidente grave": "grave",
    "com peões": "com peoes",
}
linhas = []
for nome, coluna in variaveis.items():
    tabela = pd.crosstab(ambos["ano"], ambos[coluna])
    qui2, valor_p = chi2_contingency(tabela)[:2]
    v_cramer = np.sqrt(qui2 / (tabela.values.sum() * (min(tabela.shape) - 1)))
    linhas.append({"Distribuição": nome, "V de Cramér": round(v_cramer, 3), "Valor-p": round(valor_p, 3)})

quadro4 = pd.DataFrame(linhas)
print("\nQuadro 4 - Associação entre o ano e cada distribuição\n", quadro4.to_string(index=False))
