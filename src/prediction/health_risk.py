"""
Health Risk Mapping Module
============================
Maps AQI categories to health risk levels and descriptions,
reproducing the base study's health risk scoring mechanism.
"""

import numpy as np
import pandas as pd


# Health risk mapping based on AQI categories
HEALTH_RISK_MAP = {
    0: {  # Good
        "risk_level": "Minimal",
        "risk_score": 1,
        "color": "#009966",
        "impact": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "advisory": "No precautions needed. Enjoy outdoor activities.",
    },
    1: {  # Satisfactory
        "risk_level": "Low",
        "risk_score": 2,
        "color": "#FFDE33",
        "impact": "Air quality is acceptable. Some pollutants may be of concern for a very small number of unusually sensitive individuals.",
        "advisory": "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    },
    2: {  # Moderate
        "risk_level": "Moderate",
        "risk_score": 3,
        "color": "#FF9933",
        "impact": "Members of sensitive groups may experience health effects. The general public is unlikely to be affected.",
        "advisory": "People with respiratory disease (asthma), elderly, and children should limit prolonged outdoor exertion.",
    },
    3: {  # Poor
        "risk_level": "High",
        "risk_score": 4,
        "color": "#CC0033",
        "impact": "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects.",
        "advisory": "Avoid prolonged outdoor activities. Use masks if going outside. Sensitive groups should stay indoors.",
    },
    4: {  # Very Poor
        "risk_level": "Very High",
        "risk_score": 5,
        "color": "#660099",
        "impact": "Health alert: everyone may experience more serious health effects.",
        "advisory": "Minimize outdoor activities. Use N95 masks. Keep windows closed. Use air purifiers indoors.",
    },
    5: {  # Severe
        "risk_level": "Severe",
        "risk_score": 6,
        "color": "#7E0023",
        "impact": "Health emergency: the entire population is at serious risk.",
        "advisory": "AVOID all outdoor activities. Stay indoors with air purification. Seek medical attention if experiencing symptoms.",
    },
}


def map_health_risk(aqi_code):
    """
    Map a single AQI category code to its health risk information.

    Parameters
    ----------
    aqi_code : int
        AQI category code (0-5).

    Returns
    -------
    dict
        Health risk information.
    """
    if aqi_code in HEALTH_RISK_MAP:
        return HEALTH_RISK_MAP[aqi_code].copy()
    return {"risk_level": "Unknown", "risk_score": 0, "color": "#808080",
            "impact": "Unknown", "advisory": "No information available."}


def add_health_risk_columns(df, aqi_code_col="aqi_code"):
    """
    Add health risk columns to a DataFrame based on AQI category codes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with AQI category codes.
    aqi_code_col : str
        Name of the AQI code column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added risk columns.
    """
    df_out = df.copy()

    risk_info = df_out[aqi_code_col].apply(map_health_risk)

    df_out["risk_level"] = risk_info.apply(lambda x: x["risk_level"])
    df_out["risk_score"] = risk_info.apply(lambda x: x["risk_score"])
    df_out["risk_color"] = risk_info.apply(lambda x: x["color"])
    df_out["health_impact"] = risk_info.apply(lambda x: x["impact"])
    df_out["health_advisory"] = risk_info.apply(lambda x: x["advisory"])

    print(f"[Health Risk] Added risk columns for {len(df_out)} records")
    print(f"Risk Distribution:")
    print(df_out["risk_level"].value_counts().to_string())

    return df_out


def get_risk_summary(predictions_df, aqi_code_col="predicted_aqi_code"):
    """
    Generate a summary of health risk distribution from predictions.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predicted AQI codes.
    aqi_code_col : str
        Name of the prediction column.

    Returns
    -------
    pd.DataFrame
        Summary table with risk distribution.
    """
    codes = predictions_df[aqi_code_col].value_counts().sort_index()
    summary_rows = []

    for code, count in codes.items():
        risk = HEALTH_RISK_MAP.get(code, {})
        summary_rows.append({
            "AQI Code": code,
            "Risk Level": risk.get("risk_level", "Unknown"),
            "Risk Score": risk.get("risk_score", 0),
            "Count": count,
            "Percentage": f"{100 * count / len(predictions_df):.1f}%",
        })

    return pd.DataFrame(summary_rows)
