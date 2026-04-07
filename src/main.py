import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import math
from dataclasses import dataclass
from enum import Enum


#глобальные данные
soil = [];
weather = [];
# Планируемая урожайность
target_yield = 6.0;




class Predecessor(Enum):
    """Типы предшественников"""
    WINTER_WHEAT = "озимая пшеница"
    CORN = "кукуруза"
    SUNFLOWER = "подсолнечник"
    LEGUMES = "бобовые"
    FALLOW = "чистый пар"
    POTATOES = "картофель"


@dataclass
class Soil:
    """Состояние почвы"""
    humus: float  # % (гумус)
    ph: float  # pH (KCl)
    n_mg_kg: float  # минеральный азот (нитраты + аммоний), мг/кг
    p_mg_kg: float  # подвижный фосфор (по Чирикову), мг/кг
    k_mg_kg: float  # обменный калий (по Чирикову), мг/кг
    soil_type: str = "chernozem"

@dataclass
class Weather:
    """Погодные условия вегетационного периода"""
    precipitation_mm: float  # сумма осадков, мм
    avg_temp_c: float  # средняя температура, °C


@dataclass
class PredecessorInfo:
    """Информация о предшественнике"""
    crop: Predecessor
    yield_t_ha: float  # урожайность предшественника, т/га


@dataclass
class FertilizerRates:
    """Результат: рекомендуемые дозы удобрений"""
    n_kg_ha: float  # азот (N) кг/га д.в.
    p_kg_ha: float  # фосфор (P2O5) кг/га д.в.
    k_kg_ha: float  # калий (K2O) кг/га д.в.

    def __str__(self):
        return (f"Рекомендуемые дозы удобрений:\n"
                f"  N (азот) : {self.n_kg_ha:.1f} кг/га\n"
                f"  P (фосфор): {self.p_kg_ha:.1f} кг/га\n"
                f"  K (калий) : {self.k_kg_ha:.1f} кг/га")


def calculate_fertilizer_rates(
        target_crop: str,
        target_yield_t_ha: float,
        soil: Soil,
        weather: Weather,
        predecessor: PredecessorInfo
) -> FertilizerRates:
    # ----- 1. Базовый вынос на планируемый урожай -----
    # Вынос на 1 т основной продукции (кг N/P2O5/K2O)
    # Данные для пшеницы
    if target_crop == "wheat":
        uptake_per_ton = {"n": 30.0, "p": 12.0, "k": 25.0}  # кг/т
    else:
        # по умолчанию для других
        uptake_per_ton = {"n": 25.0, "p": 10.0, "k": 20.0}

    total_uptake = {
        "n": uptake_per_ton["n"] * target_yield_t_ha,
        "p": uptake_per_ton["p"] * target_yield_t_ha,
        "k": uptake_per_ton["k"] * target_yield_t_ha,
    }

    # ----- 2. -----
    # Азот: доступность из почвы (коэффициент использования N из гумуса)
    humus_factor = min(1.2, max(0.6, 1.0 - (soil.humus - 4.0) * 0.05))

    # Фосфор: коэффициент доступности из почвы зависит от pH и содержания P
    if soil.p_mg_kg > 100:
        p_soil_supply = 0.15
    elif soil.p_mg_kg > 60:
        p_soil_supply = 0.25
    elif soil.p_mg_kg > 30:
        p_soil_supply = 0.45
    else:
        p_soil_supply = 0.65

    # Коррекция по pH (при высоком pH доступность P снижается)
    if soil.ph > 7.5:
        p_soil_supply *= 0.7
    elif soil.ph < 5.5:
        p_soil_supply *= 0.8

    # Калий: коэффициент доступности из почвы
    if soil.k_mg_kg > 200:
        k_soil_supply = 0.15
    elif soil.k_mg_kg > 120:
        k_soil_supply = 0.30
    elif soil.k_mg_kg > 70:
        k_soil_supply = 0.50
    else:
        k_soil_supply = 0.70

    # Запас элемента в почве (пересчет в кг/га, слой 0-30 см)
    soil_supply_kg = {
        "n": soil.n_mg_kg * 3.0,
        "p": soil.p_mg_kg * 3.0,
        "k": soil.k_mg_kg * 3.0,
    }

    # Доступное из почвы количество
    available_from_soil = {
        "n": soil_supply_kg["n"] * humus_factor,
        "p": soil_supply_kg["p"] * p_soil_supply,
        "k": soil_supply_kg["k"] * k_soil_supply,
    }


    #python - m jupyter notebook
    #.venv\Scripts\activate

    # Установите jupyter
    #pip install jupyter

    # Теперь запускайте
    #jupyter notebook
    # ----- 3. погодf -----
    # Азот: осадки и температура
    precip_factor = 1.0
    if weather.precipitation_mm < 200:  # засуха
        precip_factor = 0.7
    elif weather.precipitation_mm > 450:  # избыток осадков
        precip_factor = 1.25

    temp_factor = 1.0
    if weather.avg_temp_c < 12:  # холодно
        temp_factor = 0.85
    elif weather.avg_temp_c > 22:
        temp_factor = 1.1

    weather_factor_n = precip_factor * temp_factor

    # Для фосфора и калия влияние погоды слабее
    weather_factor_p = 1.0
    weather_factor_k = 1.0
    if weather.avg_temp_c < 10:
        weather_factor_p = 1.15  # холодная весна - хуже доступность P
        weather_factor_k = 0.95

    # ----- 4. Поправка на предшественник (только для азота) -----
    predecessor_factor = 1.0
    predecessor_n_credit = 0.0

    if predecessor.crop == Predecessor.LEGUMES:
        predecessor_factor = 0.65
        # Бобовые оставляют 40-80 кг/га азота
        predecessor_n_credit = 50.0
    elif predecessor.crop == Predecessor.FALLOW:
        predecessor_factor = 0.70
    elif predecessor.crop == Predecessor.WINTER_WHEAT:
        predecessor_factor = 1.05
    elif predecessor.crop == Predecessor.CORN:
        predecessor_factor = 1.00
    elif predecessor.crop == Predecessor.SUNFLOWER:
        predecessor_factor = 1.10  # сильно истощает
    elif predecessor.crop == Predecessor.POTATOES:
        predecessor_factor = 0.95

    # Учет урожайности предшественника (высокий урожай -> меньше остаточного питания)
    if predecessor.yield_t_ha > 5.0:
        predecessor_factor *= 1.05

    # ----- 5. Коэффициент использования из удобрений -----
    # Доля, которую растение реально усвоит из внесенного удобрения
    fertilizer_efficiency = {
        "n": 0.65,
        "p": 0.35,  # фосфор усваивается хуже
        "k": 0.55,
    }

    # ----- 6. Итоговый расчет доз -----
    # Необходимо внести = (Вынос - Доступно из почвы) / КПД удобрений
    # с учетом поправок

    # Азот (с учетом погоды и предшественника)
    required_n = max(0, total_uptake["n"] - available_from_soil["n"] - predecessor_n_credit)
    required_n *= predecessor_factor
    required_n *= weather_factor_n
    dose_n = required_n / fertilizer_efficiency["n"]

    # Фосфор
    required_p = max(0, total_uptake["p"] - available_from_soil["p"])
    required_p *= weather_factor_p
    dose_p = required_p / fertilizer_efficiency["p"]

    # Калий
    required_k = max(0, total_uptake["k"] - available_from_soil["k"])
    required_k *= weather_factor_k
    dose_k = required_k / fertilizer_efficiency["k"]

    # Округление и минимальные стартовые дозы
    dose_n = max(30.0, round(dose_n / 5) * 5)
    dose_p = max(15.0, round(dose_p / 5) * 5)
    dose_k = max(0.0, round(dose_k / 5) * 5)  # калий может не требоваться

    return FertilizerRates(n_kg_ha=dose_n, p_kg_ha=dose_p, k_kg_ha=dose_k)






# Загрузка CSV
def load_CSV():
    global soil
    global weather
    global predecessor
    data_Soil = pd.read_csv('soil.csv')
    data_Weather = pd.read_csv('weather.csv')
    index_Soil = 1
    index_Weather = 1
    soil = Soil(
        humus=data_Soil["humus"][index_Soil],  # %
        ph=data_Soil["ph"][index_Soil],
        n_mg_kg=data_Soil["n_mg_kg"][index_Soil],  # минеральный азот (NO3+NH4)
        p_mg_kg=data_Soil["p_mg_kg"][index_Soil],  # подвижный фосфор
        k_mg_kg=data_Soil["k_mg_kg"][index_Soil],  # обменный калий
        soil_type=data_Soil["soil_type"][index_Soil]
    )
    weather = Weather(
        precipitation_mm=data_Weather["precipitation_mm"][index_Weather],  # умеренные осадки
        avg_temp_c=data_Weather["avg_temp_c"][index_Weather]  # теплый сезон
    )
    predecessor = PredecessorInfo(
        crop=Predecessor.WINTER_WHEAT,
        yield_t_ha=4.2
    )


#Запуск программы
all_soils = load_CSV();


def predict_base_yield(soil: Soil, weather: Weather) -> float:
    """
    Прогноз базовой урожайности (без удобрений)
    на основе почвенных и погодных условий.
    """
    # Базовый потенциал от гумуса (2 т/га при 2% гумуса)
    base = 2.0 + (soil.humus - 2.0) * 0.4

    # Поправка на pH
    if soil.ph < 5.5 or soil.ph > 8.0:
        base *= 0.8
    elif soil.ph < 6.0:
        base *= 0.95

    # Поправка на содержание элементов
    base += min(2.0, soil.n_mg_kg / 40)
    base += min(1.5, soil.p_mg_kg / 100)
    base += min(1.0, soil.k_mg_kg / 200)

    # Поправка на погоду
    if weather.precipitation_mm < 200:
        base *= 0.65
    elif weather.precipitation_mm > 500:
        base *= 0.85

    if weather.avg_temp_c < 10 or weather.avg_temp_c > 30:
        base *= 0.85

    return max(1.0, min(7.0, round(base, 1)))


def predict_yield_with_fertilizer(base_yield: float, rates: FertilizerRates, target_yield: float) -> float:
    """
    Прогноз урожайности при внесении рекомендованных доз удобрений.
    """
    # Эффективность удобрений (упрощенно)
    n_effect = min(2.5, rates.n_kg_ha / 80)  # максимум +2.5 т/га
    p_effect = min(1.0, rates.p_kg_ha / 100)  # максимум +1.0 т/га
    k_effect = min(0.8, rates.k_kg_ha / 120)  # максимум +0.8 т/га

    total_effect = n_effect + p_effect + k_effect

    predicted = base_yield + total_effect
    # Не может быть намного выше плановой урожайности
    predicted = min(target_yield + 0.8, predicted)

    return round(predicted, 1)



# Расчет для пшеницы с планируемой урожайностью 6 т/га
rates = calculate_fertilizer_rates(
    target_crop="wheat",
    target_yield_t_ha=6.0,
    soil=soil,
    weather=weather,
    predecessor=predecessor
)

print(rates)
print("\n--- Детали расчета ---")
print(f"Вынос NPK для пшеницы на 6 т/га: N≈180, P≈72, K≈150 кг/га")


# ==================================================
# ПРОГНОЗ УРОЖАЙНОСТИ И ПОСТРОЕНИЕ ДИАГРАММЫ
# ==================================================

# Прогноз базовой урожайности (без удобрений)
base_yield = predict_base_yield(soil, weather)

# Прогноз урожайности с удобрениями
rec_yield = predict_yield_with_fertilizer(base_yield, rates, target_yield)

print(f"\n--- Прогноз урожайности ---")
print(f"Базовая урожайность (без удобрений): {base_yield} т/га")
print(f"Прогноз с рекомендованными удобрениями: {rec_yield} т/га")
print(f"Ожидаемая прибавка: {rec_yield - base_yield} т/га")

# Построение диаграммы
plt.figure(figsize=(8, 5))
bars = plt.bar(['Без удобрений', 'С рекомендацией'],
               [base_yield, rec_yield],
               color=['#FF6B6B', '#4ECDC4'],
               edgecolor='black',
               linewidth=1.5)

plt.ylabel('Урожайность (т/га)', fontsize=12)
plt.title('Сравнение урожайности: базовый вариант vs с удобрениями', fontsize=14, fontweight='bold')
plt.ylim(0, max(base_yield, rec_yield) + 1.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Подписи значений на столбцах
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height} т/га',
             ha='center',
             va='bottom',
             fontsize=12,
             fontweight='bold')

plt.tight_layout()
plt.savefig('results/results1.png', dpi=150)
plt.show()

# Дополнительный график: зависимость прибавки от дозы азота
plt.figure(figsize=(8, 5))

# Генерируем точки для графика (разные дозы азота)
n_doses = range(0, 301, 30)
yield_increases = []
for dose in n_doses:
    test_rates = FertilizerRates(n_kg_ha=dose, p_kg_ha=rates.p_kg_ha, k_kg_ha=rates.k_kg_ha)
    y_pred = predict_yield_with_fertilizer(base_yield, test_rates, target_yield)
    yield_increases.append(y_pred - base_yield)

plt.plot(n_doses, yield_increases, marker='o', color='green', linewidth=2, markersize=8)
plt.axvline(x=rates.n_kg_ha, color='red', linestyle='--', linewidth=2,
            label=f'Рекомендуемая доза N = {rates.n_kg_ha:.0f} кг/га')
plt.xlabel('Доза азота (N, кг/га)', fontsize=12)
plt.ylabel('Прибавка урожая (т/га)', fontsize=12)
plt.title('Эффективность внесения азотных удобрений', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/results2.png', dpi=150)
plt.show()