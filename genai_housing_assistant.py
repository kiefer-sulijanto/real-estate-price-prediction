import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import joblib
import pandas as pd
from typing import Dict, Any, Optional
from openai import OpenAI
import matplotlib.pyplot as plt

# 🔇 SUPPRESS WARNINGS
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.core")

# -----------------------------------------
# CONFIG
# -----------------------------------------

DATA_PATH = "usa_real_estate.csv"
CITY_ENCODER_PATH = "final_city_encoder.pkl"
STATE_ENCODER_PATH = "final_state_encoder.pkl"
MODEL_PATH = "best_model_pipeline.pkl"
CITY_PPSQFT_PATH = "city_avg_ppsqft.pkl"  

FEATURE_COLUMNS = ["bed", "bath", "house_size", "lot_size", "city", "state"]

BASE_YEAR = 2025            # For projections
ANNUAL_GROWTH_RATE = 0.04   # 4% per year
STATE_ABBREV = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY"
}


client = OpenAI()


# -----------------------------------------
# LOAD MODEL + ENCODERS + DATASET
# -----------------------------------------

def load_components():
    model = joblib.load(MODEL_PATH)
    city_enc = joblib.load(CITY_ENCODER_PATH)
    state_enc = joblib.load(STATE_ENCODER_PATH)
    data = pd.read_csv(DATA_PATH)
    return model, city_enc, state_enc, data


def load_city_ppsqft(path: str = CITY_PPSQFT_PATH) -> Dict[str, float]:
    """
    Load city_avg_ppsqft.pkl safely.

    Expected format (what you currently have):
        {
            "Dallas_TX": 210.5,
            "Austin_TX": 195.3,
            ...
        }

    Returns:
        dict with lowercase keys, e.g. "dallas_tx" -> avg_ppsqft
    """
    if not os.path.exists(path):
        print("⚠️ city_avg_ppsqft.pkl not found, skipping city baseline.")
        return {}

    try:
        obj = joblib.load(path)
    except Exception as e:
        print("⚠️ Failed to load city_avg_ppsqft.pkl:", e)
        return {}

    mapping: Dict[str, float] = {}

    # Case 1: dict 
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                mapping[str(k).strip().lower()] = float(v)
            except (TypeError, ValueError):
                continue
        print(f"ℹ️ Loaded city_avg_ppsqft mapping with {len(mapping)} entries.")
        return mapping

    # Case 2: DataFrame 
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        

        # pick value column = first numeric column
        num_cols = df.select_dtypes("number").columns
        if len(num_cols) == 0:
            print("⚠️ No numeric column found in city_avg_ppsqft.pkl DataFrame.")
            return {}
        value_col = num_cols[0]

        cols_lower = {c.lower(): c for c in df.columns}

        # if has separate "city" + "state" columns → combine
        if "city" in cols_lower and "state" in cols_lower:
            city_col = cols_lower["city"]
            state_col = cols_lower["state"]
            for _, row in df[[city_col, state_col, value_col]].dropna().iterrows():
                key = f"{row[city_col]}_{row[state_col]}".strip().lower()
                mapping[key] = float(row[value_col])
        else:
            # fallback: first non-numeric col as key
            non_num_cols = [c for c in df.columns if c not in num_cols]
            if not non_num_cols:
                print("⚠️ No non-numeric column to use as key in city_avg_ppsqft.pkl DataFrame.")
                return {}
            key_col = non_num_cols[0]
            for _, row in df[[key_col, value_col]].dropna().iterrows():
                key = str(row[key_col]).strip().lower()
                mapping[key] = float(row[value_col])

        print(f"ℹ️ Loaded city_avg_ppsqft mapping with {len(mapping)} entries (DataFrame).")
        return mapping

    # Case 3: Series / other
    try:
        s = pd.Series(obj)

        for k, v in s.to_dict().items():
            try:
                mapping[str(k).strip().lower()] = float(v)
            except (TypeError, ValueError):
                continue
        print(f"ℹ️ Loaded city_avg_ppsqft mapping with {len(mapping)} entries (Series-like).")
        return mapping
    except Exception as e:
        print("⚠️ Unsupported type for city_avg_ppsqft.pkl:", type(obj), "error:", e)
        return {}




# -----------------------------------------
# GENAI → Extract features
# -----------------------------------------

def extract_features_from_text(user_text: str) -> Dict[str, Any]:
    """
    Convert user description into exact JSON schema.
    """

    system_prompt = """
You MUST output ONLY a JSON object with this exact schema:

{
 "bed": int,
 "bath": float,
 "house_size": float,
 "lot_size": float,
 "city": string,
 "state": string
}
""".strip()

    prompt = f"{system_prompt}\n\nUser description:\n{user_text}\n\nJSON only."

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    raw_text = response.output_text.strip()
    

    # Remove code fences like ```json ... ```
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```[a-zA-Z0-9]*\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()

    # Extract between first { and last }
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        raw_text = raw_text[start:end+1].strip()

    try:
        features = json.loads(raw_text)
    except Exception as e:
        print("❌ JSON PARSE ERROR:", e)
        print("❌ RAW TEXT RECEIVED:", repr(raw_text))
        raise

    return features


# -----------------------------------------
# Build DataFrame for model
# -----------------------------------------
def preprocess_for_model(
    features: Dict[str, Any],
    city_enc,
    state_enc,
    city_ppsqft: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Build the feature vector the model was trained on.

    Model expects:
      - bed
      - bath
      - house_size
      - lot_size
      - city_encoded
      - state_encoded
      - ppsqft_vs_city_avg
    """

    # --- basic fields ---
    bed = int(features["bed"])
    bath = float(features["bath"])
    house_size = float(features["house_size"])
    lot_size = float(features["lot_size"])
    city = str(features["city"]).strip()
    state_raw = str(features["state"]).strip()

    city_state = f"{city}_{state_raw}"

    # --- encode CITY using column name 'city_state' ---
    tmp_city = pd.DataFrame([{"city_state": city_state}])
    city_encoded_df = city_enc.transform(tmp_city[["city_state"]])
    city_encoded = city_encoded_df.iloc[0, 0]

    # --- encode STATE ---
    tmp_state = pd.DataFrame([{"state": state_raw}])
    state_encoded_df = state_enc.transform(tmp_state[["state"]])
    state_encoded = state_encoded_df.iloc[0, 0]

    # --- compute ppsqft_vs_city_avg from city_ppsqft dict ---
    ppsqft_vs_city_avg = 0.0  # default

    if city_ppsqft:
        # normalize state name → abbrev if needed
        if len(state_raw) > 2:
            state_key = STATE_ABBREV.get(state_raw.lower(), state_raw)
        else:
            state_key = state_raw.upper()

        dict_key = f"{city.replace(' ', '')}_{state_key}".lower()
        avg_pp = city_ppsqft.get(dict_key)

        if avg_pp is not None:
            try:
                ppsqft_vs_city_avg = float(avg_pp)
            except (TypeError, ValueError):
                pass
        else:
            # fallback: mean of all city averages
            try:
                vals = list(city_ppsqft.values())
                if vals:
                    ppsqft_vs_city_avg = float(sum(vals) / len(vals))
            except Exception:
                pass

    # --- FINAL DF sent to model ---
    df = pd.DataFrame([{
        "bed": bed,
        "bath": bath,
        "house_size": house_size,
        "lot_size": lot_size,
        "city_encoded": city_encoded,
        "state_encoded": state_encoded,
        "ppsqft_vs_city_avg": ppsqft_vs_city_avg,
    }])

    return df



# -----------------------------------------
# Prediction
# -----------------------------------------

def predict_price(model, processed_df: pd.DataFrame, min_price: Optional[float] = None) -> float:
    """
    Run model.predict on processed_df, making sure that
    the columns are in EXACTLY the same order and names
    as during model.fit().
    """

    # 🔧 Align feature columns with what the model saw during training
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)


        # Debug: show any mismatch
        missing = [c for c in expected_cols if c not in processed_df.columns]
        extra = [c for c in processed_df.columns if c not in expected_cols]
        if missing:
            print("⚠️ Missing columns for model:", missing)
        if extra:
            print("⚠️ Extra columns not used by model:", extra)

        # Reorder and subset columns to EXACT order
        processed_df = processed_df.reindex(columns=expected_cols)

    # Now call the model
    raw_pred = float(model.predict(processed_df)[0])

    # 1) ensure non-negative
    adjusted = abs(raw_pred)

    # 2) ensure not below a reasonable floor (5th percentile from dataset)
    if min_price is not None:
        adjusted = max(adjusted, float(min_price))

    return adjusted



# -----------------------------------------
# SIMPLE RAG: find similar houses
# -----------------------------------------

def find_comparables(dataset: pd.DataFrame, features: Dict[str, Any]) -> pd.DataFrame:
    df = dataset.copy()

    # Filter by city and state (case-insensitive)
    if "city" in features and isinstance(features["city"], str):
        df = df[df["city"].str.lower() == features["city"].lower()]
    if "state" in features and isinstance(features["state"], str):
        df = df[df["state"].str.lower() == features["state"].lower()]

    if df.empty:
        df = dataset

    # Distance by bed / bath / size
    df["score"] = (
        abs(df["bed"] - features["bed"]) +
        abs(df["bath"] - features["bath"]) +
        abs(df["house_size"] - features["house_size"]) / 500
    )

    df = df.sort_values("score").head(5)
    return df.drop(columns=["score"], errors="ignore")


# -----------------------------------------
# Explanation via OpenAI
# -----------------------------------------

def generate_explanation(
    user_text: str,
    features: Dict[str, Any],
    predicted_price: float,
    comparables: pd.DataFrame,
    baseline_price: Optional[float] = None
) -> str:
    table = ""
    if comparables is not None and not comparables.empty:
        table = comparables[["price", "bed", "bath", "house_size", "city", "state"]].to_markdown(index=False)

    if baseline_price is None:
        baseline_line = "No reliable city-level price-per-sqft baseline was available for this city."
    else:
        baseline_line = f"City-level baseline (avg price per sqft × house size) is about ${baseline_price:,.0f}."

    prompt = f"""
User Description:
{user_text}

Extracted Features:
{json.dumps(features, indent=2)}

Model Estimated Price: ${predicted_price:,.0f}

City Baseline:
{baseline_line}

Comparable Properties (sample from dataset):
{table}

Explain in simple language:
- Whether the model estimate seems low / high relative to the city baseline and comparables.
- How features like bed, bath, size, lot, and location influence the price.
- Mention that the estimate is approximate and markets change.
2–3 short paragraphs only.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


# -----------------------------------------
# Visualization helpers
# -----------------------------------------

def plot_price_vs_size(dataset: pd.DataFrame,
                       features: Dict[str, Any],
                       predicted_price: float,
                       comparables: pd.DataFrame):
    df = dataset.copy()
    df = df.dropna(subset=["house_size", "price"])

    if len(df) > 500:
        df = df.sample(500, random_state=42)

    plt.figure()
    plt.scatter(df["house_size"], df["price"], alpha=0.3, label="All houses")

    if comparables is not None and not comparables.empty:
        plt.scatter(
            comparables["house_size"],
            comparables["price"],
            marker="s",
            label="Comparables"
        )

    plt.scatter(
        [features["house_size"]],
        [predicted_price],
        marker="*",
        s=200,
        label="Predicted house"
    )

    plt.xlabel("House size (sqft)")
    plt.ylabel("Price")
    plt.title("Price vs House Size")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparables_bar(comparables: pd.DataFrame):
    if comparables is None or comparables.empty:
        return

    comp = comparables.copy().head(10)

    labels = [f"{row['city']}, {row['state']}" for _, row in comp.iterrows()]

    plt.figure()
    plt.bar(range(len(comp)), comp["price"])
    plt.xticks(range(len(comp)), labels, rotation=45, ha="right")
    plt.ylabel("Price")
    plt.title("Comparable Properties Prices")
    plt.tight_layout()
    plt.show()


# -----------------------------------------
# Future price projections (Assume 4% Growth Anually until 2030)
# -----------------------------------------

def project_future_prices(current_price: float,
                          base_year: int = BASE_YEAR,
                          last_year: int = 2030,
                          growth_rate: float = ANNUAL_GROWTH_RATE) -> Dict[int, float]:
    projections: Dict[int, float] = {}
    for year in range(base_year, last_year + 1):
        years_ahead = year - base_year
        projected = current_price * ((1 + growth_rate) ** years_ahead)
        projections[year] = float(projected)
    return projections


def plot_future_projection(projections: Dict[int, float]):
    if not projections:
        return

    years = list(projections.keys())
    prices = list(projections.values())

    plt.figure()
    plt.plot(years, prices, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Projected price")
    plt.title("Projected House Price Over Time")
    plt.tight_layout()
    plt.show()

def build_property_context_text(
    user_text: str,
    features: Dict[str, Any],
    predicted_price: float,
    baseline_price: Optional[float],
    comparables: pd.DataFrame,
    projections: Dict[int, float],
) -> str:
    """
    Build a text context that we feed to the LLM so it can behave
    like a property agent and answer questions about this house.
    """
    comps_table = ""
    if comparables is not None and not comparables.empty:
        cols = ["price", "bed", "bath", "house_size", "city", "state"]
        cols = [c for c in cols if c in comparables.columns]
        comps_table = comparables[cols].to_markdown(index=False)

    proj_lines = "\n".join(
        [f"{year}: ${price:,.0f}" for year, price in projections.items()]
    )

    if baseline_price is None:
        baseline_line = "No reliable city-level price-per-sqft baseline was available."
    else:
        baseline_line = f"City baseline estimate: about ${baseline_price:,.0f}."

    context = f"""
User original description:
{user_text}

Extracted features:
{json.dumps(features, indent=2)}

Model estimated price: ${predicted_price:,.0f}
{baseline_line}

Comparable properties (sample from dataset):
{comps_table}

Future price projections (assuming {ANNUAL_GROWTH_RATE*100:.1f}% annual growth):
{proj_lines}
"""
    return context.strip()


def property_agent_reply(
    user_question: str,
    context_text: str,
) -> str:
    """
    Use the LLM as a property agent that answers questions
    based on the given property context.
    """

    system_prompt = """
You are a friendly, knowledgeable real estate agent in the United States.
You are chatting with a user about ONE specific property.

Use ONLY the context provided (features, price estimate, comparables, projections).
Do NOT make up specific numbers that are not in the context.
If something is uncertain, say it is an estimate.

Your style:
- clear and conversational
- 2–4 short paragraphs max
- if user asks for advice (buy/sell/negotiate), give practical tips
""".strip()

    prompt = f"""
{system_prompt}

PROPERTY CONTEXT:
{context_text}

USER QUESTION:
{user_question}

Now respond as the property agent.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text.strip()



# -----------------------------------------
# MAIN CLI APP
# -----------------------------------------

def main():
    print("\n🏠 GenAI House Price Assistant (Pipeline + Encoder + City Baseline)")
    print("Type 'quit' to exit.\n")

    model, city_enc, state_enc, dataset = load_components()
    city_ppsqft = load_city_ppsqft()
    print("✅ Model, encoders, dataset, and city baseline loaded (if available).\n")

    # Use 5th percentile as minimum reasonable price floor
    min_price = dataset["price"].quantile(0.05)
    print("DEBUG min_price from dataset (5th percentile):", min_price)

    while True:
        user_text = input("Describe the house: ").strip()
        if user_text.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 👋")
            break

        try:
            # 1) Extract features from text (LLM)
            features = extract_features_from_text(user_text)

            # 2) Build model input
            df_ready = preprocess_for_model(features, city_enc, state_enc, city_ppsqft)
            print("DEBUG df_ready columns:", df_ready.columns.tolist())
            print(df_ready.head())

            # 3) Predict price (clamped)
            price = predict_price(model, df_ready, min_price=min_price)

            # 4) Comparables
            comparables = find_comparables(dataset, features)

            # 5) City baseline from ppsqft
            baseline_price = None
            if city_ppsqft and features.get("city") and features.get("state") and features.get("house_size"):
                city_name = str(features["city"]).strip()
                state_raw = str(features["state"]).strip()

                # Convert state full name → abbreviation if needed
                if len(state_raw) > 2:
                    state_key = STATE_ABBREV.get(state_raw.lower(), state_raw)
                else:
                    state_key = state_raw.upper()

                dict_key = f"{city_name.replace(' ', '')}_{state_key}".lower()
                avg_pp = city_ppsqft.get(dict_key)
                if avg_pp is not None:
                    baseline_price = float(avg_pp) * float(features["house_size"])

            # 6) Explanation (first answer)
            explanation = generate_explanation(
                user_text=user_text,
                features=features,
                predicted_price=price,
                comparables=comparables,
                baseline_price=baseline_price,
            )

            # 7) Output summary once
            print("\n================ RESULT ================")
            print(f"🏷️ Predicted Price: ${price:,.0f}\n")
            if baseline_price is not None:
                print(f"🏙️ City Baseline Estimate: ${baseline_price:,.0f}\n")
            print(explanation)
            print("========================================\n")

            # 8) Visualizations
            plot_price_vs_size(dataset, features, price, comparables)
            plot_comparables_bar(comparables)

            # 9) Projections
            projections = project_future_prices(price)
            print("📈 Projected prices (assuming "
                  f"{ANNUAL_GROWTH_RATE*100:.1f}% growth per year):")
            for year, proj_price in projections.items():
                print(f"  {year}: ${proj_price:,.0f}")
            print()
            plot_future_projection(projections)

            # 🔹 10) Build context for chatbot
            context_text = build_property_context_text(
                user_text=user_text,
                features=features,
                predicted_price=price,
                baseline_price=baseline_price,
                comparables=comparables,
                projections=projections,
            )

            # 🔹 11) Start chatbot loop for this property
            print("💬 You can now chat with the AI property agent about THIS house.")
            print("    Type 'back' to describe a new house, or 'quit' to exit.\n")

            while True:
                follow_up = input("User: ").strip()
                if follow_up.lower() in ["back"]:
                    print("\n⬅️  Going back to describe a new house.\n")
                    break
                if follow_up.lower() in ["quit", "exit", "q"]:
                    print("Goodbye! 👋")
                    return

                agent_answer = property_agent_reply(follow_up, context_text)
                print("\nAgent:", agent_answer, "\n")

        except Exception as e:
            import traceback
            print("⚠️ Error:", e)
            traceback.print_exc()
            print("Fix the issue above.\n")




if __name__ == "__main__":
    main()
