import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def main():

    print("[*] Carregando o dataset consolidado...")

    df = pd.read_csv("consolidated_traffic_data.csv")

    # Remove espaços invisíveis que costumam vir nos nomes das colunas do CIC
    df.columns = df.columns.str.strip()

    print("[*] Tratando anomalias matemáticas (Infinito e Nulos)...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    #Previne que a IA aprenda IPs ou portas.
    colunas_identificadoras = ['Flow ID', 'Src IP', 'Dst IP', 'Source IP', 'Destination IP', 
                               'Timestamp', 'Src Port', 'Dst Port']
    
    # Remove as colunas apenas se elas existirem no CSV
    df = df.drop(columns=[col for col in colunas_identificadoras if col in df.columns])

    # Tira a coluna de "gabarito" do dataset, colocando o que são dados para o aprendizado em X
    # Em Y fica guardado o gabarito do tipo de tráfego
    X = df.drop("traffic_type", axis=1)
    y = df["traffic_type"]

    # Traduz texto em binário
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Salva o dicionário para poder saber qual número é qual categoria
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"[*] Categorias mapeadas: {label_map}")

    #Treino e Teste de ML
    print("[*] Dividindo os dados (80% Treino, 20% Teste)...")
    # stratify=y_enc garante que a proporção de cada categoria se mantenha
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    #Treinamento do Modelo
    print("[*] Iniciando o treinamento da Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,  # Cria 300 árvores de decisão
        max_depth=30,      # Limita a profundidade para evitar decorar os dados (overfitting)
        random_state=42,   # Garante que o resultado seja reprodutível
        n_jobs=-1          # Usa todos os núcleos da CPU disponíveis
    )
    rf.fit(X_train, y_train)

    #Avaliação de desempenho
    print("\n[*] Avaliando o modelo com os dados de teste...")
    y_pred = rf.predict(X_test)
    
    print(f"\nAcurácia Geral: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n=== Relatório Detalhado ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    
if __name__ == "__main__":
    main()