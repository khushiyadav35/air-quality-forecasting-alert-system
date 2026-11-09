"""
Official Indian Central Pollution Control Board (CPCB) NAQI Breakpoints
Reference: CPCB Guidelines for Calculation of Air Quality Index (2014)
Author: Khushi Yadav (@khushiyadav35)
"""

# Official Indian NAQI Categories and Color Palette
NAQI_CATEGORIES = [
    {
        "category": "Good",
        "aqi_min": 0,
        "aqi_max": 50,
        "color": "#00E400",       # Emerald Green
        "hex_badge": "#10B981",
        "emoji": "🟢",
        "health_impact": "Minimal impact. Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "advisory": "Ideal air quality for all outdoor sports, jogging, and ventilation.",
    },
    {
        "category": "Satisfactory",
        "aqi_min": 51,
        "aqi_max": 100,
        "color": "#92D050",       # Light Green / Chartreuse
        "hex_badge": "#84CC16",
        "emoji": "🟡",
        "health_impact": "Minor breathing discomfort to sensitive people (asthmatics, elderly, infants).",
        "advisory": "Unusually sensitive individuals should consider reducing heavy prolonged outdoor exertion.",
    },
    {
        "category": "Moderate",
        "aqi_min": 101,
        "aqi_max": 200,
        "color": "#FFFF00",       # Yellow
        "hex_badge": "#EAB308",
        "emoji": "🟠",
        "health_impact": "Breathing discomfort to people with lungs, asthma, and heart diseases.",
        "advisory": "Children and adults with respiratory conditions should wear basic filtration masks near highways.",
    },
    {
        "category": "Poor",
        "aqi_min": 201,
        "aqi_max": 300,
        "color": "#FF9900",       # Orange
        "hex_badge": "#F97316",
        "emoji": "🔴",
        "health_impact": "Breathing discomfort to most people on prolonged exposure; significant aggravation of heart/lung disease.",
        "advisory": "Everyone should wear N95/FFP2 masks outdoors; avoid strenuous morning jogs in Delhi/Lucknow smog hours.",
    },
    {
        "category": "Very Poor",
        "aqi_min": 301,
        "aqi_max": 400,
        "color": "#FF0000",       # Red
        "hex_badge": "#EF4444",
        "emoji": "🟣",
        "health_impact": "Respiratory illness to the people on prolonged exposure; pronounced effect on people with existing disease.",
        "advisory": "Stay indoors as much as possible; run HEPA air purifiers; seal windows during morning inversion peaks.",
    },
    {
        "category": "Severe",
        "aqi_min": 401,
        "aqi_max": 500,
        "color": "#990000",       # Dark Crimson / Maroon
        "hex_badge": "#7F1D1D",
        "emoji": "🟤",
        "health_impact": "Affects healthy people and seriously impacts those with existing medical conditions; public health emergency.",
        "advisory": "EMERGENCY: Halt all outdoor physical activity; implement school closures / GRAP Stage-IV protocols.",
    },
]

# CPCB Pollutant Breakpoint Matrix
# Breakpoints format: (B_lo, B_hi, I_lo, I_hi)
# Note: PM2.5, PM10, NO2, SO2, NH3, Pb are in ug/m3 (24h avg)
# CO is in mg/m3 (8h avg)
# O3 is in ug/m3 (8h avg)
CPCB_BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 600.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2000.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
    "nh3": [
        (0.0, 200.0, 0, 50),
        (200.1, 400.0, 51, 100),
        (400.1, 800.0, 101, 200),
        (801.1, 1200.0, 201, 300),
        (1201.1, 1800.0, 301, 400),
        (1801.1, 2400.0, 401, 500),
    ],
}
