import os
import time
from smolagents import CodeAgent, LiteLLMModel, Tool
import osmnx as ox
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import scrolledtext
import time

# Finaliza a resposta da ferramenta
def final_answer(value: str) -> str:
   return value

#def final_answer(value: str) -> dict:
 #   return {"final_answer": value}

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 300

# Haversine: distância entre dois nós (para A* e proximidade de parques)
def haversine_distance(u, v, G):
    y1, x1 = G.nodes[u]['y'], G.nodes[u]['x']
    y2, x2 = G.nodes[v]['y'], G.nodes[v]['x']
    R = 6371000  # metros

    phi1, phi2 = radians(y1), radians(y2)
    d_phi = radians(y2 - y1)
    d_lambda = radians(x2 - x1)

    a = sin(d_phi/2)**2 + cos(phi1)*cos(phi2)*sin(d_lambda/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

# Geocodificação com fallback
def geocode_robusto(local: str) -> tuple:
    try:
        return ox.geocode(f"{local}, Belo Horizonte", timeout=10)
    except Exception:
        try:
            gdf = ox.geocode_to_gdf(f"{local}, Belo Horizonte")
            if not gdf.empty:
                pt = gdf.iloc[0].geometry.centroid
                return (pt.y, pt.x)
            else:
                raise ValueError(f"Local '{local}' não encontrado nem com fallback.")
        except Exception:
            raise ValueError(f"Local '{local}' não foi encontrado.")

def extrair_ruas(rota, G):
    ruas = []
    for u, v in zip(rota[:-1], rota[1:]):
        data = G.get_edge_data(u, v)
        if data:
            # Se houver múltiplas arestas entre os mesmos nós, pegue a primeira
            edge = data[0] if isinstance(data, dict) else data
            nome = edge.get('name')
            if isinstance(nome, list):
                ruas.append(nome[0])
            elif nome:
                ruas.append(nome)
    # Remova duplicadas mantendo a ordem
    ruas_unicas = []
    for rua in ruas:
        if rua and rua not in ruas_unicas:
            ruas_unicas.append(rua)
    return ruas_unicas


def plotar_rota_com_parques(G, rota, parques, origem, destino):
    fig, ax = ox.plot_graph_route(
        G,
        route=rota,
        route_color='blue',
        route_linewidth=3,
        node_size=0,
        bgcolor='white',
        show=False,
        close=False,
        figsize=(12, 12)
    )

    # Destaca os parques com pontos vermelhos
    for nome in parques:
        try:
            node = BHGraphManager().parks[nome]
            x = G.nodes[node]['x']
            y = G.nodes[node]['y']
            ax.scatter(x, y, c='red', s=100, label=nome)
        except:
            continue

    ax.set_title(f"Rota de {origem} até {destino}", fontsize=16)
    ax.legend(loc='lower right')

    # Salva a imagem ao invés de mostrar
    plt.savefig("rota.png", dpi=300, bbox_inches='tight')
    plt.close()



class BHGraphManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("⏳ Carregando grafo de BH (aguarde ~1 minuto na primeira execução)...")
        start_time = time.time()

        custom_filter = '["highway"~"primary|secondary|tertiary"]'
        self.G = ox.graph_from_place(
            "Belo Horizonte, MG, Brazil",
            network_type="walk",
            custom_filter=custom_filter,
            simplify=True
        )

        tags = {"leisure": "park"}
        features = ox.features_from_place("Belo Horizonte, MG, Brazil", tags=tags)

        #Filtros que precisaram ser feitos na mao 
        self.parks = {
            name: ox.distance.nearest_nodes(self.G, X=geom.representative_point().x, Y=geom.representative_point().y)
            for name, geom in zip(features["name"], features["geometry"])
            if isinstance(name, str)
            and "praça" not in name.lower()
            and "quadra" not in name.lower()
            and "academia" not in name.lower()
            and "campo" not in name.lower()
            and "bosque" not in name.lower()
            and "coreto" not in name.lower()
            and "Coreto" not in name.lower()
            and "varanda urbana" not in name.lower()
            and "Varanda urbana" not in name.lower()
        }

        print(f"✅ Grafo carregado em {time.time()-start_time:.2f}s | Nós: {len(self.G.nodes)} | Arestas: {len(self.G.edges)}")
        print(f"✅ Parques mapeados: {list(self.parks.keys())}")

class RouteTool(Tool):
    name = "find_route"
    description = """
    ATENÇÃO: Esta ferramenta JÁ retorna a resposta final completa e formatada.
    O modelo NÃO deve tentar modificar, reformatar, reordenar ou resumir o conteúdo retornado.
    Simplesmente exiba o valor retornado como resposta final ao usuário.
    O modelo NÃO deve tentar acessar chaves como `.get()`, `.description`, etc.
    Apenas exiba a resposta retornada diretamente ao usuário.

    Ferramenta para calcular rotas entre dois pontos em Belo Horizonte.
    Ferramenta para calcular rotas entre dois pontos em Belo Horizonte.
    Retorna os caminhos pelos algoritmos Dijkstra e A*, além de listar os parques próximos à rota.
    Suporta comandos como 'qual parque está no caminho entre Savassi e UFMG'.
    Suporta comando como "quais parque estao entre o Centro e a UFMG"
    Suporta comando como "qual parque esta no caminho entre o Buritis e a UFMG

    Esta ferramenta já retorna a resposta final formatada.
    O modelo NÃO deve tentar reinterpretar, acessar chaves, ou aplicar filtros à resposta. Use o conteúdo retornado diretamente como resposta final ao usuário.

    """
    inputs = {
        "source": {"type": "string", "description": "Local de origem (bairro, ponto turístico, etc.)"},
        "target": {"type": "string", "description": "Local de destino (bairro, parque, etc.)"}
    }
    output_type = "string"

    def __init__(self):
        self.bh = BHGraphManager()
        self.is_initialized = True

    def forward(self, source: str, target: str) -> str:
        try:
            start_time = time.time()
            source = source.lower().replace("bairro", "").strip().title()
            target = target.lower().replace("bairro", "").strip().title()

            try:
                source_coord = geocode_robusto(source)
            except Exception:
                return final_answer(f"ERRO: Local de origem '{source}' não foi encontrado no mapa.")

            try:
                target_coord = geocode_robusto(target)
            except Exception:
                return final_answer(f"ERRO: Local de destino '{target}' não foi encontrado no mapa.")

            source_node = ox.distance.nearest_nodes(self.bh.G, X=source_coord[1], Y=source_coord[0])
            target_node = ox.distance.nearest_nodes(self.bh.G, X=target_coord[1], Y=target_coord[0])

            route_dijkstra = nx.shortest_path(self.bh.G, source=source_node, target=target_node, weight='length', method='dijkstra')
            route_astar = nx.astar_path(
                self.bh.G,
                source=source_node,
                target=target_node,
                weight='length',
                heuristic=lambda u, v: haversine_distance(u, v, self.bh.G)
            )

            ruas_dijkstra = extrair_ruas(route_dijkstra, self.bh.G)

            dist_dijkstra = nx.path_weight(self.bh.G, route_dijkstra, weight='length')
            dist_astar = nx.path_weight(self.bh.G, route_astar, weight='length')

            def esta_proximo(no_parque, rota, G, limite=200):
                for no_rota in rota:
                    if haversine_distance(no_parque, no_rota, G) <= limite:
                        return True
                return False

            parques_na_rota = [
                nome for nome, no in self.bh.parks.items()
                if esta_proximo(no, route_dijkstra, self.bh.G) or esta_proximo(no, route_astar, self.bh.G)
            ]

            plotar_rota_com_parques(self.bh.G, route_dijkstra, parques_na_rota, source, target)

            velocidade_media_m_s = 1.4  # caminhada humana (~5 km/h)
            tempo_estimado = dist_dijkstra / velocidade_media_m_s  # em segundos

            # formatar para minutos
            minutos = int(tempo_estimado // 60)
            segundos = int(tempo_estimado % 60)


            return final_answer(
                f"🗺️ Rota de '{source}' para '{target}':\n"
                f"• Dijkstra: {dist_dijkstra:.1f} metros\n"
                f"• Ruas no percurso: {', '.join(ruas_dijkstra) or 'Indisponível'}\n"
                f"• A*: {dist_astar:.1f} metros\n"
                f"• Parques no caminho: {', '.join(parques_na_rota) or 'Nenhum'}\n"
                f"• Estimativa de tempo de caminhada: {minutos} min {segundos} s \n"
                f"⏱️ Tempo total de caminhamento no grafo: {time.time()-start_time:.2f}s"
            )




        except Exception as e:
            return final_answer(f" Erro inesperado: {str(e)}")

class NearestParkTool(Tool):
    name = "nearest_park"
    description = """
    Ferramenta para identificar o parque mais próximo de um ponto em Belo Horizonte.
    Use quando o usuário perguntar algo como "qual é o parque mais próximo da UFMG" ou "tem parque perto do bairro X?".
    """

    inputs = {
        "place": {"type": "string", "description": "Nome do local (bairro, ponto turístico, etc.)"}
    }
    output_type = "string"

    def __init__(self):
        self.bh = BHGraphManager()
        self.is_initialized = True  

    def forward(self, place: str) -> str:
        try:
            place_coord = geocode_robusto(place)
        except Exception:
            return final_answer(f"ERRO: Local '{place}' não foi encontrado no mapa.")

        G = self.bh.G
        place_node = ox.distance.nearest_nodes(G, X=place_coord[1], Y=place_coord[0])

        menor_dist = float('inf')
        parque_mais_proximo = None
        parque_node = None

        for nome, no in self.bh.parks.items():
            dist = haversine_distance(place_node, no, G)
            if dist < menor_dist:
                menor_dist = dist
                parque_mais_proximo = nome
                parque_node = no

        if not parque_mais_proximo:
            return final_answer(f"ERRO: Nenhum parque foi encontrado próximo a '{place}'.")

        # Calcular rota e ruas até o parque
        try:
            rota = nx.shortest_path(G, source=place_node, target=parque_node, weight='length')
            dist_total = nx.path_weight(G, rota, weight='length')
            ruas = extrair_ruas(rota, G)

            # Tempo estimado de caminhada (~1.4 m/s)
            tempo_seg = dist_total / 1.4
            minutos = int(tempo_seg // 60)
            segundos = int(tempo_seg % 60)

            return final_answer(
                f"O parque mais próximo de *{place}* é o *{parque_mais_proximo}*, a cerca de {dist_total:.1f} metros.\n"
                f"Ruas no caminho: {', '.join(ruas) or 'Indisponível'}\n"
                f"Estimativa de tempo de caminhada: {minutos} min {segundos} s"
            )

        except Exception as e:
            return final_answer(
                f"O parque mais próximo de *{place}* é o *{parque_mais_proximo}*, a aproximadamente {menor_dist:.1f} metros.\n"
                f"(Não foi possível calcular a rota detalhada: {str(e)})"
            )


# Modelo configurado com Qwen2:7b local
model = LiteLLMModel(
    model_id="ollama/qwen2:7b",
    api_base="http://localhost:11434",
    temperature=0.3,
    timeout=180
)

agent = CodeAgent(
    tools=[RouteTool(), NearestParkTool()],
    model=model
)

# Janela interativa tkinter

def enviar_pergunta():
    pergunta = entrada.get()
    if not pergunta.strip():
        return

    saida.insert(tk.END, f"Você: {pergunta}\n")
    entrada.delete(0, tk.END)
    saida.insert(tk.END, "🔍 Processando...\n")
    janela.update()

    try:
        inicio = time.time()
        resposta = agent.run(pergunta)
        fim = time.time()

        if isinstance(resposta, dict):
            texto = resposta.get("final_answer", str(resposta))
        else:
            texto = str(resposta)

        saida.insert(tk.END, f"Agente ({fim - inicio:.2f}s): {texto}\n\n")
        saida.see(tk.END)
    except Exception as e:
        saida.insert(tk.END, f"Erro: {e}\n")

# GUI
janela = tk.Tk()
janela.title("Agente de Rotas - Belo Horizonte")

entrada = tk.Entry(janela, width=80)
entrada.pack(pady=5)

botao = tk.Button(janela, text="Enviar", command=enviar_pergunta)
botao.pack()

saida = scrolledtext.ScrolledText(janela, width=100, height=25)
saida.pack(padx=10, pady=10)

janela.mainloop()