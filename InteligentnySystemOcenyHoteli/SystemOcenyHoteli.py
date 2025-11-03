import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
"""
Zmienne wejściowe w zakresie od 0 do 10. 
Mamy trzy zmienne wejściowe czystość, komfort i cene z czego cena 10 oznacza,że jest drogo
"""
czystosc = ctrl.Antecedent(np.arange(0, 11, 1), 'czystosc')
komfort = ctrl.Antecedent(np.arange(0, 11, 1), 'komfort')
cena = ctrl.Antecedent(np.arange(0, 11, 1), 'cena')

"""
Zmienne wyjściowa w zakresie od 1 do 5. Symbolizuje ilość wiazdek
"""

gwiazdki = ctrl.Consequent(np.arange(1, 6, 0.1), 'gwiazdki')

"""
Funkcje przynależności dla czystosci i komfortu jest to "niska,średnia,wysoka" 
osobno dla ceny jest "tania,średnia,droga"
"""

for var in [czystosc, komfort]:
    var['niska'] = fuzz.trimf(var.universe, [0, 0, 5])
    var['srednia'] = fuzz.trimf(var.universe, [3, 5, 7])
    var['wysoka'] = fuzz.trimf(var.universe, [5, 10, 10])

cena['tania'] = fuzz.trimf(cena.universe, [0, 0, 5])
cena['srednia'] = fuzz.trimf(cena.universe, [3, 5, 7])
cena['droga'] = fuzz.trimf(cena.universe, [5, 10, 10])

gwiazdki['1'] = fuzz.trimf(gwiazdki.universe, [1, 1, 2])
gwiazdki['2'] = fuzz.trimf(gwiazdki.universe, [1, 2, 3])
gwiazdki['3'] = fuzz.trimf(gwiazdki.universe, [2, 3, 4])
gwiazdki['4'] = fuzz.trimf(gwiazdki.universe, [3, 4, 5])
gwiazdki['5'] = fuzz.trimf(gwiazdki.universe, [4, 5, 5])

"""
Reguły na podstawie jakich logika przydziela przynależność zmiennych wejściowych do zmiennej wyjściowej 
"""

rules = [
    ctrl.Rule(czystosc['wysoka'] & komfort['wysoka'] & cena['tania'], gwiazdki['5']),
    ctrl.Rule(czystosc['srednia'] & komfort['srednia'], gwiazdki['3']),
    ctrl.Rule((czystosc['niska'] | komfort['niska']) & (cena['srednia'] | cena['droga']) , gwiazdki['1']),
    ctrl.Rule(cena['tania'] & komfort['srednia'] , gwiazdki['4']),
    ctrl.Rule(cena['droga'] & (komfort['srednia'] | czystosc['srednia']), gwiazdki['2']),
]

"""
pierwsza linijka rworzy zbiór reguł dla naszego fuzzy logic
druga linijka odpowiada za stworzenie symulacji dla której podajemy dane wejściowe
"""

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

"""
Do pierwszych trzech linijek podajemy dane wejśiowe dla każej zmiennej
a następnie prowadzimy obliczenia
"""

sim.input['czystosc'] = 8
sim.input['komfort'] = 5
sim.input['cena'] = 0
sim.compute()

"""
pierwsza linijka zraca jaką ocenę gwiazdkową wyliczył system z podanych danych
druga zwraca graf obrazujący przynależność odpowiedzi na podstawie danych
"""

print(f"Ocena końcowa (gwiazdki): {sim.output['gwiazdki']:.2f}")
gwiazdki.view(sim=sim)