import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pandas_datareader.data as web 
import streamlit as st
import matplotlib.pyplot as plt

# ======================================================================
# 0. CONFIGURACIÓN DEL SITIO WEB
# ======================================================================

st.set_page_config(
    page_title="DEYCO - Score de Riesgo Lineal V2.0",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================================
# 1. CONFIGURACIÓN FINAL DEL MODELO V2.0 (Fijas)
# ======================================================================

PONDERACIONES_FINALES = {
    "Momentum_SPY": 30, "HY_Spread": 19, "MOVE_Index": 12,
    "FCI_Endurecimiento": 9, "Dolar_Fuerte_DXY": 6, "Tasa_Morosidad": 5,
    "M2_Crecimiento_YoY": 3, "GEX_Agregado": 0.0, "FGI_Sentimiento": 0.0
}
MAX_SCORE = sum(PONDERACIONES_FINALES.values())
UMBRAL_FINAL = 70 

# Fechas Fijas
START_DATE_FIXED = '2007-01-01' # Usaré 2007-01-01 para incluir la GFC, es más informativo
END_DATE_FIXED = datetime.now().strftime('%Y-%m-%d')

# Umbrales (Necesarios para el cálculo del score)
UMBRALES = {
    "Momentum_SPY": [-5.00, 0.00, False], "HY_Spread": [6.00, 3.00, True],
    "MOVE_Index": [150.00, 75.00, True], "GEX_Agregado": [-1.0, 1.0, False],
    "FCI_Endurecimiento": [1.0, -1.0, True], "Tasa_Morosidad": [5.00, 3.00, True],
    "Dolar_Fuerte_DXY": [110.00, 90.00, True], "M2_Crecimiento_YoY": [0.0, 5.00, False],
    "FGI_Sentimiento": [20.00, 80.00, False]
}

# ======================================================================
# 2. FUNCIONES DE CÁLCULO
# ======================================================================

@st.cache_data
def calculate_metrics(df: pd.DataFrame, equity_col: str, daily_return_col: str, trading_days: int=252) -> dict:
    """Calcula las métricas de rendimiento."""
    if len(df) < 2: return {'CAGR': 0, 'Sharpe': 0, 'MDD': 0, 'Volatilidad': 0, 'Tiempo Inv.': 0}
    
    total_days = (df.index[-1] - df.index[0]).days
    cagr = ((df[equity_col].iloc[-1] / df[equity_col].iloc[0]) ** (365.25 / total_days)) - 1
    volatility = df[daily_return_col].std() * np.sqrt(trading_days)
    peak = df[equity_col].expanding(min_periods=1).max()
    drawdown = (df[equity_col] / peak) - 1
    mdd = drawdown.min()
    sharpe_ratio = cagr / volatility if volatility != 0 else 0
    time_invested = (df['Investment_Signal'].sum() / len(df)) * 100 if 'Investment_Signal' in df.columns else 100.0
    
    return {
        'CAGR': cagr, 'Sharpe': sharpe_ratio, 'MDD': mdd, 
        'Volatilidad': volatility, 'Tiempo Inv.': time_invested
    }

def calcular_score_componente(valor: float, nombre_comp: str, ponderacion: float) -> float:
    """Calcula la contribución de un componente con la ponderación final."""
    P_MAX = ponderacion
    if P_MAX == 0: return 0.0
    U_PELIGRO, U_SEGURIDAD, RIESGO_ALTO = UMBRALES.get(nombre_comp, [0, 0, False])
    rango = abs(U_PELIGRO - U_SEGURIDAD)
    if rango == 0: return P_MAX
    
    if RIESGO_ALTO:
        distancia_norm = (U_PELIGRO - valor) / rango
    else: 
        distancia_norm = (valor - U_PELIGRO) / rango
        
    contribucion_norm = max(0.0, min(1.0, distancia_norm))
    return contribucion_norm * P_MAX

def calcular_score_total(row: pd.Series, ponderaciones_actuales: Dict[str, float]) -> int:
    """Calcula el score total con las ponderaciones finales."""
    score = 0.0
    for comp, weight in ponderaciones_actuales.items():
        if weight > 0 and comp in row.index:
            valor_escalar = row[comp].item() if hasattr(row[comp], 'item') else row[comp]
            score += calcular_score_componente(valor_escalar, comp, weight)
    
    return round(min(score, MAX_SCORE))

def interpretar_score_actual(score: int, max_score: int) -> tuple:
    """Traduce el score numérico a un estado operativo (Semáforo)."""
    if score >= 70:
        estado = "BAJO RIESGO/COMPRA (Go)"
        mensaje = f"✅ El Score de Riesgo DEYCO está en ZONA VERDE. El entorno es favorable para la Renta Variable (Score: {score}/{max_score})."
        color = "green"
    elif score >= 56:
        estado = "RIESGO ALTO (Espera)"
        mensaje = f"⚠️ El Score de Riesgo DEYCO está en ZONA AMARILLA. Se recomienda MANTENERSE EN CAJA o ser cauteloso (Score: {score}/{max_score})."
        color = "orange"
    else: 
        estado = "RIESGO EXTREMO (Stop)"
        mensaje = f"❌ El Score de Riesgo DEYCO está en ZONA ROJA. La gestión de riesgo exige ESTAR FUERA DEL MERCADO (Score: {score}/{max_score})."
        color = "red"
    return estado, mensaje, color

# ======================================================================
# 3. OBTENCIÓN Y PREPARACIÓN DE DATOS (Con caché)
# ======================================================================

@st.cache_data
def obtener_datos_historicos_final(start_date: str, end_date: str) -> pd.DataFrame:
    
    TICKER_SPX = '^GSPC'
    # Descarga de datos
    data = yf.download(TICKER_SPX, start=start_date, end=end_date, progress=False)
    if data.empty: raise ValueError("No se obtuvieron datos válidos del S&P 500.")
    
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1) 
    df = data[['Close']].rename(columns={'Close': 'SPX_Price'}).copy()
    df['Daily_Return'] = df['SPX_Price'].pct_change()
    df['Momentum_SPY'] = (df['SPX_Price'].pct_change(periods=5) * 100)
    df.index = pd.to_datetime(df.index) 
    
    # Descarga de indicadores (simplificado para el script web)
    tickers_yf = {'DX-Y.NYB': 'Dolar_Fuerte_DXY', '^MOVE': 'MOVE_Index'}
    tickers_fred = {'WM2NS': 'M2_Crecimiento_YoY_BASE', 'BAA': 'BAA', 'AAA': 'AAA', 'DRALACBS': 'Tasa_Morosidad', 'NFCI': 'FCI_Endurecimiento'}

    for ticker, name in tickers_yf.items():
        data_yf = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data_yf.empty and 'Close' in data_yf.columns:
            if isinstance(data_yf.columns, pd.MultiIndex): data_yf.columns = data_yf.columns.droplevel(1) 
            df = df.join(data_yf['Close'].rename(name), how='left')
    
    for ticker, name in tickers_fred.items():
        try:
            data_fred = web.DataReader(ticker, 'fred', start=start_date)
            df = df.join(data_fred.rename(columns={ticker: name}), how='left')
        except Exception: pass 
            
    df = df.ffill()
    df['M2_Crecimiento_YoY'] = df['M2_Crecimiento_YoY_BASE'].pct_change(periods=12) * 100
    df['HY_Spread'] = df['BAA'] - df['AAA']
    
    required_keys = [k for k, v in PONDERACIONES_FINALES.items() if v > 0]
    df = df.dropna(subset=['SPX_Price', 'Daily_Return'] + required_keys).fillna(0)
    
    return df.drop(columns=['M2_Crecimiento_YoY_BASE', 'BAA', 'AAA'], errors='ignore')

# ======================================================================
# 4. FUNCIÓN PRINCIPAL DE LA APLICACIÓN
# ======================================================================

def main():
    
    st.title("DEYCO: Score de Riesgo Lineal V2.0")
    st.markdown(f"**Periodo de Backtest Fijo:** `{START_DATE_FIXED}` a `{END_DATE_FIXED}`")

    try:
        with st.spinner('Cargando datos históricos y ejecutando backtest...'):
            df_data = obtener_datos_historicos_final(START_DATE_FIXED, END_DATE_FIXED)
    except ValueError as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return

    # 1. Calcular Score y Señal
    df_data['Risk_Score'] = df_data.apply(
        lambda row: calcular_score_total(row, PONDERACIONES_FINALES), axis=1
    )
    df_data['Investment_Signal'] = np.where(df_data['Risk_Score'] >= UMBRAL_FINAL, 1, 0)
    
    # 2. Rendimiento
    df_data['Strategy_Return'] = df_data['Daily_Return'] * df_data['Investment_Signal']
    df_data['B&H_Return'] = df_data['Daily_Return']
    initial_capital = 100
    df_data['Strategy_Equity'] = initial_capital * (1 + df_data['Strategy_Return']).cumprod()
    df_data['B&H_Equity'] = initial_capital * (1 + df_data['B&H_Return']).cumprod()

    # ======================================================================
    # PUNTO 2: Mostrar el score para el día con comentarios
    # ======================================================================
    st.header("1. Señal Operativa Actual")
    
    ultimo_score = df_data['Risk_Score'].iloc[-1]
    estado_operativo, mensaje_operativo, color = interpretar_score_actual(ultimo_score, MAX_SCORE)
    
    # Mostrar el score con un formato de caja grande
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 10px; color: white;">
            <h3 style="margin: 0; text-align: center;">{estado_operativo}</h3>
            <p style="margin: 0; text-align: center; font-size: 1.2em;">{mensaje_operativo}</p>
        </div>
        """, unsafe_allow_html=True
    )

    # ======================================================================
    # PUNTO 3: Mostrar la gráfica comparativa (DEYCO vs SPX)
    # ======================================================================
    st.header("2. Curva de Capital (DEYCO vs. S&P 500)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gráfica del Equity
    ax.plot(df_data['Strategy_Equity'], label="DEYCO (Score V2.0)", color='green', linewidth=2)
    ax.plot(df_data['B&H_Equity'], label="S&P 500 (Buy & Hold)", color='red', linestyle='--', linewidth=1.5)
    
    # Formato de la gráfica
    ax.set_title(f'Rendimiento Comparativo (Base 100)', fontsize=16)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Equity (Valor Absoluto)')
    ax.legend(loc='upper left')
    ax.grid(True, which="both", ls="--", c='0.7')
    ax.ticklabel_format(style='plain', axis='y')
    
    st.pyplot(fig)
    
    # ======================================================================
    # PUNTO 4: Cuadro de Métricas (ROI, MDD, etc.)
    # ======================================================================
    st.header("3. Métricas de Rendimiento")
    
    metrics_deyco = calculate_metrics(df_data, 'Strategy_Equity', 'Strategy_Return')
    metrics_spx = calculate_metrics(df_data.assign(Investment_Signal=1), 'B&H_Equity', 'B&H_Return')
    
    # Crear la tabla de métricas
    data_metrics = {
        'Métrica': ['CAGR Anualizado (ROI)', 'MDD Máximo', 'Sharpe Ratio', 'Volatilidad Anual', 'Tiempo Invertido'],
        'S&P 500 (B&H)': [
            f"{metrics_spx['CAGR']*100:.2f}%", 
            f"{metrics_spx['MDD']*100:.2f}%", 
            f"{metrics_spx['Sharpe']:.2f}", 
            f"{metrics_spx['Volatilidad']*100:.2f}%", 
            f"{metrics_spx['Tiempo Inv.']:.2f}%"
        ],
        'DEYCO (Score V2.0)': [
            f"{metrics_deyco['CAGR']*100:.2f}%", 
            f"{metrics_deyco['MDD']*100:.2f}%", 
            f"{metrics_deyco['Sharpe']:.2f}", 
            f"{metrics_deyco['Volatilidad']*100:.2f}%", 
            f"{metrics_deyco['Tiempo Inv.']:.2f}%"
        ]
    }
    df_metrics = pd.DataFrame(data_metrics).set_index('Métrica')
    
    st.table(df_metrics)


if __name__ == "__main__":
    main()