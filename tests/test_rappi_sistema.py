"""
==============================================================
  TESTS DE CALIDAD (QA) — SISTEMA RAPPI LIMA
  Cubre: dijkstra, bfs, dfs, calcular_metricas_ruta
  Complejidad Algorítmica 1ACC0184 - 2026-10
==============================================================
"""

import unittest
import time
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rappi_algoritmos import dijkstra, bfs, dfs, calcular_metricas_ruta


# ─────────────────────────────────────────────────────────────
#  FIXTURES COMPARTIDOS
# ─────────────────────────────────────────────────────────────

def grafo_lineal():
    """
    Grafo lineal simple: 0 → 1 → 2 → 3
    Pesos conocidos para validar cálculos exactos.
    """
    return {
        0: [(1, {"tiempo_min": 5.0, "distancia_km": 1.2, "costo_sol": 3.5})],
        1: [(2, {"tiempo_min": 3.0, "distancia_km": 0.8, "costo_sol": 2.0})],
        2: [(3, {"tiempo_min": 7.0, "distancia_km": 2.1, "costo_sol": 5.0})],
        3: [],
    }


def grafo_bifurcado():
    """
    Grafo con dos caminos entre 0 y 3:
      Ruta A (rápida en tiempo): 0→1→3  tiempo=4, dist=3.0
      Ruta B (corta en paradas): 0→2→3  tiempo=10, dist=1.5
    Sirve para verificar que Dijkstra elige por peso
    y BFS elige por número de saltos (ambas tienen 2 saltos → BFS puede elegir cualquiera).
    """
    return {
        0: [
            (1, {"tiempo_min": 2.0, "distancia_km": 1.5, "costo_sol": 2.0}),
            (2, {"tiempo_min": 6.0, "distancia_km": 0.8, "costo_sol": 1.5}),
        ],
        1: [(3, {"tiempo_min": 2.0, "distancia_km": 1.5, "costo_sol": 2.0})],
        2: [(3, {"tiempo_min": 4.0, "distancia_km": 0.7, "costo_sol": 1.0})],
        3: [],
    }


def grafo_desconectado():
    """
    Dos componentes separadas: {0,1,2} y {5,6}
    """
    return {
        0: [(1, {"tiempo_min": 3.0, "distancia_km": 1.0, "costo_sol": 2.0})],
        1: [(2, {"tiempo_min": 2.0, "distancia_km": 0.5, "costo_sol": 1.0})],
        2: [],
        5: [(6, {"tiempo_min": 1.0, "distancia_km": 0.3, "costo_sol": 0.5})],
        6: [],
    }


def grafo_con_ciclo():
    """
    Grafo con ciclo: 0→1→2→1 (ciclo entre 1 y 2), salida en 3
    Verifica que los algoritmos no entren en bucle infinito.
    """
    return {
        0: [(1, {"tiempo_min": 2.0, "distancia_km": 0.5, "costo_sol": 1.0})],
        1: [(2, {"tiempo_min": 3.0, "distancia_km": 1.0, "costo_sol": 2.0}),
            (3, {"tiempo_min": 10.0, "distancia_km": 4.0, "costo_sol": 8.0})],
        2: [(1, {"tiempo_min": 3.0, "distancia_km": 1.0, "costo_sol": 2.0}),
            (3, {"tiempo_min": 1.0, "distancia_km": 0.3, "costo_sol": 0.5})],
        3: [],
    }


def df_nodos_mock():
    """DataFrame mínimo para calcular_metricas_ruta."""
    return pd.DataFrame([
        {"nodo_id": 0, "pedidos_hora": 20},
        {"nodo_id": 1, "pedidos_hora": 5},
        {"nodo_id": 2, "pedidos_hora": 0},
        {"nodo_id": 3, "pedidos_hora": 15},
    ])


# ─────────────────────────────────────────────────────────────
#  TESTS — DIJKSTRA
# ─────────────────────────────────────────────────────────────

class TestDijkstra(unittest.TestCase):

    def setUp(self):
        self.t0 = time.perf_counter()

    def tearDown(self):
        ms = (time.perf_counter() - self.t0) * 1000
        print(f"    {ms:.4f} ms")

    # ── Correctitud básica ────────────────────────────────────

    def test_ruta_optima_por_tiempo(self):
        """Dijkstra encuentra el camino de menor tiempo en grafo lineal."""
        print("\n[Dijkstra] ruta_optima_por_tiempo", end="  ")
        dist, camino, _ = dijkstra(grafo_lineal(), 0, 3, peso="tiempo_min")
        self.assertEqual(camino, [0, 1, 2, 3])
        self.assertAlmostEqual(dist, 15.0)

    def test_ruta_optima_por_distancia(self):
        """Dijkstra puede optimizar por distancia_km además de tiempo."""
        print("\n[Dijkstra] ruta_optima_por_distancia", end="  ")
        dist, camino, _ = dijkstra(grafo_lineal(), 0, 3, peso="distancia_km")
        self.assertEqual(camino, [0, 1, 2, 3])
        self.assertAlmostEqual(dist, 4.1)

    def test_ruta_optima_por_costo(self):
        """Dijkstra puede optimizar por costo_sol."""
        print("\n[Dijkstra] ruta_optima_por_costo", end="  ")
        dist, camino, _ = dijkstra(grafo_lineal(), 0, 3, peso="costo_sol")
        self.assertEqual(camino, [0, 1, 2, 3])
        self.assertAlmostEqual(dist, 10.5)

    def test_elige_ruta_rapida_en_bifurcacion(self):
        """Con dos caminos Dijkstra elige el de menor tiempo, no el de menos paradas."""
        print("\n[Dijkstra] elige_ruta_rapida_en_bifurcacion", end="  ")
        dist, camino, _ = dijkstra(grafo_bifurcado(), 0, 3, peso="tiempo_min")
        self.assertEqual(camino, [0, 1, 3])
        self.assertAlmostEqual(dist, 4.0)

    def test_origen_igual_destino(self):
        """Si origen == destino el costo es 0 y el camino contiene solo ese nodo."""
        print("\n[Dijkstra] origen_igual_destino", end="  ")
        dist, camino, _ = dijkstra(grafo_lineal(), 2, 2)
        self.assertEqual(dist, 0.0)
        self.assertIn(2, camino)

    # ── Robustez / casos extremos ─────────────────────────────

    def test_nodo_destino_inexistente(self):
        """Destino que no existe en el grafo retorna infinito y camino vacío."""
        print("\n[Dijkstra] nodo_destino_inexistente", end="  ")
        dist, camino, _ = dijkstra(grafo_lineal(), 0, 999)
        self.assertEqual(dist, float("inf"))
        self.assertEqual(camino, [])

    def test_nodo_origen_inexistente(self):
        """Origen que no existe en el grafo retorna infinito y camino vacío."""
        print("\n[Dijkstra] nodo_origen_inexistente", end="  ")
        dist, camino, _ = dijkstra(grafo_lineal(), 999, 3)
        self.assertEqual(dist, float("inf"))
        self.assertEqual(camino, [])

    def test_grafo_desconectado(self):
        """No hay ruta entre componentes distintas: retorna infinito."""
        print("\n[Dijkstra] grafo_desconectado", end="  ")
        dist, camino, _ = dijkstra(grafo_desconectado(), 0, 5)
        self.assertEqual(dist, float("inf"))
        self.assertEqual(camino, [])

    def test_no_entra_en_bucle_con_ciclo(self):
        """En grafo con ciclo Dijkstra termina y encuentra la ruta óptima."""
        print("\n[Dijkstra] no_entra_en_bucle_con_ciclo", end="  ")
        dist, camino, _ = dijkstra(grafo_con_ciclo(), 0, 3, peso="tiempo_min")
        self.assertNotEqual(dist, float("inf"))
        self.assertEqual(camino[0], 0)
        self.assertEqual(camino[-1], 3)

    def test_camino_es_continuo(self):
        """Cada par consecutivo del camino existe como arista en el grafo."""
        print("\n[Dijkstra] camino_es_continuo", end="  ")
        g = grafo_lineal()
        _, camino, _ = dijkstra(g, 0, 3)
        for u, v in zip(camino[:-1], camino[1:]):
            vecinos = [w for w, _ in g.get(u, [])]
            self.assertIn(v, vecinos, f"Arista {u}→{v} no existe en el grafo")

    def test_visitados_no_repite_nodos(self):
        """La lista de visitados no contiene nodos duplicados."""
        print("\n[Dijkstra] visitados_no_repite_nodos", end="  ")
        _, _, visitados = dijkstra(grafo_lineal(), 0, 3)
        self.assertEqual(len(visitados), len(set(visitados)))

    def test_grafo_vacio(self):
        """Grafo vacío retorna infinito y camino vacío sin lanzar excepción."""
        print("\n[Dijkstra] grafo_vacio", end="  ")
        dist, camino, _ = dijkstra({}, 0, 1)
        self.assertEqual(dist, float("inf"))
        self.assertEqual(camino, [])


# ─────────────────────────────────────────────────────────────
#  TESTS — BFS
# ─────────────────────────────────────────────────────────────

class TestBFS(unittest.TestCase):

    def setUp(self):
        self.t0 = time.perf_counter()

    def tearDown(self):
        ms = (time.perf_counter() - self.t0) * 1000
        print(f"    {ms:.4f} ms")

    def test_encuentra_ruta_existente(self):
        """BFS encuentra camino válido cuando existe ruta."""
        print("\n[BFS] encuentra_ruta_existente", end="  ")
        camino, _ = bfs(grafo_lineal(), 0, 3)
        self.assertTrue(len(camino) > 0)
        self.assertEqual(camino[0], 0)
        self.assertEqual(camino[-1], 3)

    def test_menor_numero_de_paradas(self):
        """BFS garantiza el camino con menos aristas (paradas)."""
        print("\n[BFS] menor_numero_de_paradas", end="  ")
        # En el grafo lineal, la única ruta es 0→1→2→3 (3 aristas)
        camino, _ = bfs(grafo_lineal(), 0, 3)
        self.assertEqual(len(camino), 4)  # 4 nodos = 3 aristas

    def test_origen_igual_destino(self):
        """BFS con origen == destino retorna ese nodo sin explorar más."""
        print("\n[BFS] origen_igual_destino", end="  ")
        camino, visitados = bfs(grafo_lineal(), 1, 1)
        self.assertEqual(camino, [1])

    def test_destino_inalcanzable(self):
        """BFS retorna camino vacío cuando no hay ruta al destino."""
        print("\n[BFS] destino_inalcanzable", end="  ")
        camino, _ = bfs(grafo_desconectado(), 0, 5)
        self.assertEqual(camino, [])

    def test_nodo_inexistente(self):
        """BFS retorna camino vacío para destino que no existe en el grafo."""
        print("\n[BFS] nodo_inexistente", end="  ")
        camino, _ = bfs(grafo_lineal(), 0, 999)
        self.assertEqual(camino, [])

    def test_camino_es_continuo(self):
        """Cada par consecutivo del camino BFS existe como arista."""
        print("\n[BFS] camino_es_continuo", end="  ")
        g = grafo_lineal()
        camino, _ = bfs(g, 0, 3)
        for u, v in zip(camino[:-1], camino[1:]):
            vecinos = [w for w, _ in g.get(u, [])]
            self.assertIn(v, vecinos)

    def test_no_repite_nodos_en_camino(self):
        """El camino BFS no pasa dos veces por el mismo nodo."""
        print("\n[BFS] no_repite_nodos_en_camino", end="  ")
        camino, _ = bfs(grafo_con_ciclo(), 0, 3)
        self.assertEqual(len(camino), len(set(camino)))

    def test_grafo_vacio(self):
        """BFS sobre grafo vacío retorna camino vacío sin excepción."""
        print("\n[BFS] grafo_vacio", end="  ")
        camino, _ = bfs({}, 0, 1)
        self.assertEqual(camino, [])


# ─────────────────────────────────────────────────────────────
#  TESTS — DFS
# ─────────────────────────────────────────────────────────────

class TestDFS(unittest.TestCase):

    def setUp(self):
        self.t0 = time.perf_counter()

    def tearDown(self):
        ms = (time.perf_counter() - self.t0) * 1000
        print(f"   {ms:.4f} ms")

    def test_encuentra_ruta_existente(self):
        """DFS encuentra alguna ruta cuando existe conectividad."""
        print("\n[DFS] encuentra_ruta_existente", end="  ")
        camino, _ = dfs(grafo_lineal(), 0, 3)
        self.assertTrue(len(camino) > 0)
        self.assertEqual(camino[0], 0)
        self.assertEqual(camino[-1], 3)

    def test_destino_inalcanzable(self):
        """DFS retorna camino vacío cuando no hay ruta."""
        print("\n[DFS] destino_inalcanzable", end="  ")
        camino, _ = dfs(grafo_desconectado(), 0, 5)
        self.assertEqual(camino, [])

    def test_nodo_inexistente(self):
        """DFS retorna camino vacío para destino inexistente."""
        print("\n[DFS] nodo_inexistente", end="  ")
        camino, _ = dfs(grafo_lineal(), 0, 999)
        self.assertEqual(camino, [])

    def test_no_entra_en_bucle_con_ciclo(self):
        """DFS termina correctamente en grafo con ciclos."""
        print("\n[DFS] no_entra_en_bucle_con_ciclo", end="  ")
        camino, visitados = dfs(grafo_con_ciclo(), 0, 3)
        # No importa si encontró ruta; lo crítico es que termine
        self.assertIsInstance(camino, list)
        self.assertIsInstance(visitados, list)

    def test_respeta_max_depth(self):
        """DFS con max_depth=2 no puede alcanzar nodo a 3 saltos."""
        print("\n[DFS] respeta_max_depth", end="  ")
        # 0→1→2→3 requiere 3 saltos; con max_depth=2 no llega a 3
        camino, _ = dfs(grafo_lineal(), 0, 3, max_depth=2)
        self.assertEqual(camino, [])

    def test_max_depth_suficiente(self):
        """DFS con max_depth suficiente sí alcanza el destino."""
        print("\n[DFS] max_depth_suficiente", end="  ")
        camino, _ = dfs(grafo_lineal(), 0, 3, max_depth=40)
        self.assertEqual(camino[-1], 3)

    def test_no_repite_nodos_visitados(self):
        """DFS no visita el mismo nodo dos veces."""
        print("\n[DFS] no_repite_nodos_visitados", end="  ")
        _, visitados = dfs(grafo_lineal(), 0, 3)
        self.assertEqual(len(visitados), len(set(visitados)))

    def test_grafo_vacio(self):
        """DFS sobre grafo vacío retorna camino vacío sin excepción.
        El nodo origen puede aparecer en visitados (comportamiento esperado del DFS:
        marca el nodo como visitado antes de buscar vecinos, aunque no haya ninguno)."""
        print("\n[DFS] grafo_vacio", end="  ")
        camino, visitados = dfs({}, 0, 1)
        self.assertEqual(camino, [])
        self.assertIsInstance(visitados, list)  # no lanza excepción


# ─────────────────────────────────────────────────────────────
#  TESTS — CALCULAR_METRICAS_RUTA
# ─────────────────────────────────────────────────────────────

class TestCalcularMetricasRuta(unittest.TestCase):

    def setUp(self):
        self.grafo  = grafo_lineal()
        self.df     = df_nodos_mock()
        self.t0     = time.perf_counter()

    def tearDown(self):
        ms = (time.perf_counter() - self.t0) * 1000
        print(f"    {ms:.4f} ms")

    def test_metricas_ruta_completa(self):
        """Métricas acumuladas correctas para ruta 0→1→2→3."""
        print("\n[Metricas] ruta_completa", end="  ")
        m = calcular_metricas_ruta([0, 1, 2, 3], self.grafo, self.df)
        self.assertAlmostEqual(m["tiempo_min"],   15.0)
        self.assertAlmostEqual(m["distancia_km"],  4.1)
        self.assertAlmostEqual(m["costo_sol"],     10.5)
        self.assertEqual(m["n_paradas"], 4)

    def test_metricas_ruta_parcial(self):
        """Métricas correctas para ruta parcial 0→1→2."""
        print("\n[Metricas] ruta_parcial", end="  ")
        m = calcular_metricas_ruta([0, 1, 2], self.grafo, self.df)
        self.assertAlmostEqual(m["tiempo_min"],   8.0)
        self.assertAlmostEqual(m["distancia_km"], 2.0)
        self.assertAlmostEqual(m["costo_sol"],    5.5)
        self.assertEqual(m["n_paradas"], 3)

    def test_metricas_camino_vacio(self):
        """Camino vacío retorna todas las métricas en cero."""
        print("\n[Metricas] camino_vacio", end="  ")
        m = calcular_metricas_ruta([], self.grafo, self.df)
        self.assertEqual(m["distancia_km"], 0)
        self.assertEqual(m["tiempo_min"],   0)
        self.assertEqual(m["costo_sol"],    0)
        self.assertEqual(m["n_paradas"],    0)

    def test_metricas_un_solo_nodo(self):
        """Camino de un solo nodo retorna métricas en cero y n_paradas=1."""
        print("\n[Metricas] un_solo_nodo", end="  ")
        m = calcular_metricas_ruta([0], self.grafo, self.df)
        self.assertEqual(m["distancia_km"], 0)
        self.assertEqual(m["n_paradas"],    1)

    def test_pedidos_sumados_correctamente(self):
        """Los pedidos_hora se acumulan de todos los nodos del camino."""
        print("\n[Metricas] pedidos_sumados", end="  ")
        m = calcular_metricas_ruta([0, 1, 2, 3], self.grafo, self.df)
        # nodo 0: 20, nodo 1: 5, nodo 2: 0, nodo 3: 15 → total 40
        self.assertEqual(m["pedidos"], 40)

    def test_valores_son_redondeados(self):
        """Las métricas numéricas se devuelven redondeadas (max 3 decimales)."""
        print("\n[Metricas] valores_redondeados", end="  ")
        m = calcular_metricas_ruta([0, 1, 2, 3], self.grafo, self.df)
        for campo in ("distancia_km", "tiempo_min", "costo_sol"):
            val = m[campo]
            self.assertEqual(val, round(val, 3),
                             f"{campo} no está redondeado a 3 decimales")

    def test_retorna_todas_las_claves(self):
        """El dict de métricas contiene exactamente las 5 claves esperadas."""
        print("\n[Metricas] retorna_todas_las_claves", end="  ")
        m = calcular_metricas_ruta([0, 1], self.grafo, self.df)
        esperadas = {"distancia_km", "tiempo_min", "costo_sol", "pedidos", "n_paradas"}
        self.assertEqual(set(m.keys()), esperadas)


# ─────────────────────────────────────────────────────────────
#  TESTS — INTEGRACIÓN (flujo completo)
# ─────────────────────────────────────────────────────────────

class TestIntegracion(unittest.TestCase):
    """
    Simula el flujo real de rappi_main.py:
    grafo → elegir par → correr los 3 algoritmos → calcular métricas
    """

    def setUp(self):
        self.grafo = grafo_lineal()
        self.df    = df_nodos_mock()
        self.t0    = time.perf_counter()

    def tearDown(self):
        ms = (time.perf_counter() - self.t0) * 1000
        print(f"    {ms:.4f} ms")

    def test_dijkstra_siempre_mejor_o_igual_que_bfs_en_tiempo(self):
        """Dijkstra nunca produce mayor tiempo que BFS (Dijkstra es óptimo en peso)."""
        print("\n[Integración] dijkstra_mejor_o_igual_que_bfs_en_tiempo", end="  ")
        t_dijk, _, _ = dijkstra(self.grafo, 0, 3, peso="tiempo_min")
        camino_bfs, _ = bfs(self.grafo, 0, 3)
        m_bfs = calcular_metricas_ruta(camino_bfs, self.grafo, self.df)
        self.assertLessEqual(t_dijk, m_bfs["tiempo_min"] + 1e-9)

    def test_tres_algoritmos_llegan_al_mismo_destino(self):
        """Dijkstra, BFS y DFS deben encontrar rutas que terminen en el destino."""
        print("\n[Integración] tres_algoritmos_llegan_al_mismo_destino", end="  ")
        _, c_d, _ = dijkstra(self.grafo, 0, 3)
        c_b, _    = bfs(self.grafo, 0, 3)
        c_f, _    = dfs(self.grafo, 0, 3)
        for algo, camino in [("Dijkstra", c_d), ("BFS", c_b), ("DFS", c_f)]:
            self.assertEqual(camino[-1], 3, f"{algo} no terminó en el destino")

    def test_metricas_consistentes_con_camino_dijkstra(self):
        """Las métricas calculadas sobre el camino Dijkstra son positivas y coherentes."""
        print("\n[Integración] metricas_consistentes_con_dijkstra", end="  ")
        _, camino, _ = dijkstra(self.grafo, 0, 3)
        m = calcular_metricas_ruta(camino, self.grafo, self.df)
        self.assertGreater(m["tiempo_min"],   0)
        self.assertGreater(m["distancia_km"], 0)
        self.assertGreater(m["costo_sol"],    0)
        self.assertGreaterEqual(m["pedidos"], 0)

    def test_grafo_con_ciclo_no_falla_ninguno(self):
        """Los tres algoritmos terminan sin excepción en grafo con ciclos."""
        print("\n[Integración] grafo_con_ciclo_no_falla_ninguno", end="  ")
        g = grafo_con_ciclo()
        try:
            dijkstra(g, 0, 3)
            bfs(g, 0, 3)
            dfs(g, 0, 3)
        except Exception as e:
            self.fail(f"Un algoritmo lanzó excepción inesperada: {e}")


# ─────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  PRUEBAS DE CALIDAD (QA) — SISTEMA RAPPI LIMA 2026")
    print("  Dijkstra · BFS · DFS · Métricas · Integración")
    print("="*62)
    unittest.main(verbosity=2)
