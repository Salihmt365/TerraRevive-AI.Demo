# ============================================================
#  Post-conflict farmland rehabilitation - PROTOTYPE DEMO
#  Run with:  python3 -m streamlit run app.py
# ============================================================
#  This first version is RULE-BASED: it compares soil values
#  with reference thresholds and builds a plan from rules.
#  An AI model will be trained later, once field data exists.
# ============================================================

import math
import random
import time
import unicodedata
from datetime import date

import folium
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from fpdf import FPDF
from streamlit_folium import st_folium

APP_NAME = "TerraRevive"  # <-- change this to your project name

# ------------------------------------------------------------
# 1) REFERENCE DATA
# ------------------------------------------------------------
# Heavy metal reference values in mg/kg (dry soil).
# Source: Dutch soil remediation "target" and "intervention" values
# (indicative only - have an expert verify them before real use).
#   below target        -> normal
#   target-intervention -> elevated (caution)
#   above intervention  -> seriously contaminated
METALS = {
    "Lead (Pb)":     {"target": 85,  "intervention": 530, "plants": ["Indian mustard (Brassica juncea)", "Sunflower (Helianthus annuus)"]},
    "Cadmium (Cd)":  {"target": 0.8, "intervention": 12,  "plants": ["Indian mustard (Brassica juncea)", "Sunflower (Helianthus annuus)"]},
    "Zinc (Zn)":     {"target": 140, "intervention": 720, "plants": ["Indian mustard (Brassica juncea)", "Sunflower (Helianthus annuus)"]},
    "Copper (Cu)":   {"target": 36,  "intervention": 190, "plants": ["Indian mustard (Brassica juncea)", "Sunflower (Helianthus annuus)"]},
    "Nickel (Ni)":   {"target": 35,  "intervention": 210, "plants": ["Indian mustard (Brassica juncea)"]},
    "Chromium (Cr)": {"target": 100, "intervention": 380, "plants": ["Indian mustard (Brassica juncea)", "Sunflower (Helianthus annuus)"]},
    "Arsenic (As)":  {"target": 29,  "intervention": 55,  "plants": ["Chinese brake fern (Pteris vittata)"]},
}

# Example scenarios so the demo can be shown quickly (SIMULATED data).
SCENARIOS = {
    "Scenario A - Mostly clean field": {
        "metals": {"Lead (Pb)": 40, "Cadmium (Cd)": 0.4, "Zinc (Zn)": 90, "Copper (Cu)": 25,
                   "Nickel (Ni)": 20, "Chromium (Cr)": 60, "Arsenic (As)": 10},
        "sensors": {"moisture": 22, "ph": 7.1, "ec": 1.2, "n": 25, "p": 14, "k": 180},
    },
    "Scenario B - Lead & cadmium contamination": {
        "metals": {"Lead (Pb)": 320, "Cadmium (Cd)": 4.5, "Zinc (Zn)": 260, "Copper (Cu)": 45,
                   "Nickel (Ni)": 30, "Chromium (Cr)": 90, "Arsenic (As)": 18},
        "sensors": {"moisture": 12, "ph": 5.4, "ec": 2.1, "n": 12, "p": 6, "k": 95},
    },
    "Scenario C - Severe contamination": {
        "metals": {"Lead (Pb)": 780, "Cadmium (Cd)": 14, "Zinc (Zn)": 900, "Copper (Cu)": 210,
                   "Nickel (Ni)": 60, "Chromium (Cr)": 150, "Arsenic (As)": 62},
        "sensors": {"moisture": 9, "ph": 8.1, "ec": 4.8, "n": 8, "p": 4, "k": 70},
    },
}

DEFAULT_COUNTRY = "SY"     # Syria
DEFAULT_PLACE = "169389"   # Idlib (GeoNames id)


# ------------------------------------------------------------
# 2) WORLD LOCATIONS (GeoNames: every place with 500+ people)
# ------------------------------------------------------------
# places.csv.gz and countries.csv come from the GeoNames database.
# They are stored in a compact file so the app starts fast and uses little memory.
from collections import namedtuple
import csv
import gzip
from pathlib import Path

Place = namedtuple("Place", ["name", "lat", "lon", "country"])
DATA_DIR = Path(__file__).parent


@st.cache_resource
def load_world():
    with open(DATA_DIR / "countries.csv", encoding="utf-8") as f:
        countries = {row["code"]: row["name"] for row in csv.DictReader(f)}
    places = {}   # country code -> list of place ids, biggest place first
    by_id = {}    # place id -> Place
    with gzip.open(DATA_DIR / "places.csv.gz", "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for pid, country, name, lat, lon in reader:
            by_id[pid] = Place(name, float(lat), float(lon), country)
            places.setdefault(country, []).append(pid)
    return countries, places, by_id


COUNTRIES, PLACES, PLACE_BY_ID = load_world()
COUNTRY_CODES = sorted([c for c in COUNTRIES if c in PLACES], key=lambda c: COUNTRIES[c])


def place_label(pid):
    p = PLACE_BY_ID[pid]
    # coordinates are shown so places with the same name can be told apart
    return f"{p.name}  ({p.lat:.2f}, {p.lon:.2f})"


# ------------------------------------------------------------
# 3) PUBLIC HAZARD DATA (OpenStreetMap)
# ------------------------------------------------------------
# Official mine-action databases (e.g. IMSMA) are not public, so the demo
# checks OpenStreetMap, where volunteers and organizations map known
# minefields (hazard=minefield) and military danger areas.
# This data is INCOMPLETE: "nothing found" never means "safe".
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HAZARD_RADIUS_M = 3000


def distance_km(lat1, lon1, lat2, lon2):
    """Distance between two points on Earth (haversine formula)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def point_in_polygon(lat, lon, poly):
    """True if the point is inside the polygon (list of (lat, lon))."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def parse_hazards(data, lat, lon):
    """Turn an Overpass answer into a list of hazard areas with their distance to the field."""
    hazards = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = "Minefield" if tags.get("hazard") == "minefield" else "Military danger area"
        shapes = []
        if "geometry" in el:                                   # a way
            shapes.append([(g["lat"], g["lon"]) for g in el["geometry"]])
        for m in el.get("members", []):                        # a multipolygon relation
            if "geometry" in m:
                shapes.append([(g["lat"], g["lon"]) for g in m["geometry"]])
        if "lat" in el:                                        # a single point
            shapes.append([(el["lat"], el["lon"])])
        shapes = [s for s in shapes if s]
        if not shapes:
            continue
        dist = min(distance_km(lat, lon, a, b) for s in shapes for a, b in s)
        if any(len(s) > 2 and point_in_polygon(lat, lon, s) for s in shapes):
            dist = 0.0
        hazards.append({"kind": kind, "name": tags.get("name", ""), "distance_km": round(dist, 2),
                        "shapes": shapes, "osm_id": f"{el['type']}/{el['id']}"})
    return sorted(hazards, key=lambda h: h["distance_km"])


@st.cache_data(ttl=3600, show_spinner=False)
def query_overpass(lat, lon, radius_m):
    """Ask OpenStreetMap for mapped hazard areas near the field (results are cached for 1 hour)."""
    query = f"""
    [out:json][timeout:20];
    (
      nwr["hazard"="minefield"](around:{radius_m},{lat},{lon});
      way["military"="danger_area"](around:{radius_m},{lat},{lon});
      relation["military"="danger_area"](around:{radius_m},{lat},{lon});
    );
    out geom;
    """
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=15,
                         headers={"User-Agent": f"{APP_NAME}-prototype"})
    resp.raise_for_status()
    return resp.json()


def fetch_hazards(lat, lon, radius_m=HAZARD_RADIUS_M):
    """Returns (list of hazards, error message or None). Never crashes the app.
    To stay fast, data is downloaded once per ~5 km area (with a wider radius) and
    reused for every click inside that area; distances are then computed locally."""
    cell_lat, cell_lon = round(lat * 20) / 20, round(lon * 20) / 20
    failed = st.session_state.setdefault("hazard_failures", {})
    if time.time() - failed.get((cell_lat, cell_lon), 0) < 300:   # failed recently -> don't wait again
        return [], "Public hazard data could not be checked right now (server unavailable)."
    try:
        data = query_overpass(cell_lat, cell_lon, radius_m + 5000)
        hazards = [h for h in parse_hazards(data, lat, lon) if h["distance_km"] * 1000 <= radius_m]
        return hazards, None
    except Exception as e:  # no internet, server busy, etc. (retried after 5 minutes)
        failed[(cell_lat, cell_lon)] = time.time()
        return [], f"Public hazard data could not be checked right now ({type(e).__name__})."


# ------------------------------------------------------------
# 4) ANALYSIS FUNCTIONS (the "rule engine")
# ------------------------------------------------------------
def safety_status(on_record, saw_ordnance, cleared, nearest_hazard_km=None):
    """Decide if it is safe to work on the field."""
    if cleared == "Yes":
        return "CLEARED", "The field has been surveyed or cleared. You can continue."
    if nearest_hazard_km is not None and nearest_hazard_km < 1:
        where = "covers your field" if nearest_hazard_km == 0 else f"is {nearest_hazard_km} km from your field"
        return "HIGH RISK", (f"A mapped hazard area {where}. Do NOT enter or work on this field. "
                             "Contact a mine-action / demining organization.")
    if on_record == "Yes" or saw_ordnance == "Yes":
        return "HIGH RISK", ("Do NOT enter or work on this field. Report it to a mine-action / "
                             "demining organization and wait for clearance.")
    if on_record == "Not sure" or saw_ordnance == "Not sure":
        return "UNCONFIRMED", ("Risk is unknown. Ask local authorities or a demining organization "
                               "to check the field before entering it.")
    return "LOW RISK", "No known risk. Stay alert and stop work if you see anything suspicious."


def metal_status(name, value):
    ref = METALS[name]
    if value > ref["intervention"]:
        return "Seriously contaminated"
    if value > ref["target"]:
        return "Elevated"
    return "Normal"


def analyse_metals(values):
    rows = []
    for name, value in values.items():
        ref = METALS[name]
        rows.append({
            "Metal": name,
            "Measured (mg/kg)": value,
            "Target value": ref["target"],
            "Intervention value": ref["intervention"],
            "% of intervention value": round(100 * value / ref["intervention"]),
            "Status": metal_status(name, value),
        })
    return pd.DataFrame(rows)


def overall_status(metal_df):
    """The worst status among all metals."""
    for level in ["Seriously contaminated", "Elevated"]:
        if (metal_df["Status"] == level).any():
            return level
    return "Normal"


def analyse_sensors(s):
    """Return a list of (parameter, value, finding) for sensor data. Thresholds are indicative."""
    out = []
    if s["ph"] < 5.5:
        out.append(("pH", s["ph"], "Strongly acidic - metals move more easily into plants"))
    elif s["ph"] < 6.0:
        out.append(("pH", s["ph"], "Slightly acidic"))
    elif s["ph"] <= 7.5:
        out.append(("pH", s["ph"], "Good range"))
    else:
        out.append(("pH", s["ph"], "Alkaline"))

    out.append(("Moisture (%)", s["moisture"],
                "Dry - drought risk" if s["moisture"] < 15 else "Adequate"))
    out.append(("Salinity EC (dS/m)", s["ec"],
                "Saline - salt stress for most crops" if s["ec"] > 4 else "Normal"))
    out.append(("Nitrogen N (mg/kg)", s["n"], "Low" if s["n"] < 20 else "Adequate"))
    out.append(("Phosphorus P (mg/kg)", s["p"], "Low" if s["p"] < 10 else "Adequate"))
    out.append(("Potassium K (mg/kg)", s["k"], "Low" if s["k"] < 100 else "Adequate"))
    return out


def build_plan(metal_df, s):
    """Build a step-by-step rehabilitation plan as a list of (phase title, [actions])."""
    plan = []
    elevated = metal_df[metal_df["Status"] != "Normal"]
    severe = metal_df[metal_df["Status"] == "Seriously contaminated"]

    # Phase 1 - contamination
    actions = []
    if len(severe) > 0:
        actions.append("Do NOT grow food or animal feed on this field until levels drop. "
                       "Seriously contaminated: " + ", ".join(severe["Metal"]) + ".")
        actions.append("Contact agricultural or environmental authorities: plants alone may be too slow "
                       "for this level, and professional remediation may be needed.")
    if len(elevated) > 0:
        plants = []
        for metal in elevated["Metal"]:
            for p in METALS[metal]["plants"]:
                if p not in plants:
                    plants.append(p)
        actions.append("Phytoremediation: plant " + " and ".join(plants) +
                       " to draw metals out of the soil. Expect several growing seasons.")
        actions.append("WARNING: harvested remediation plants are contaminated. Never eat them or feed them "
                       "to animals. Dispose of them safely or send them for bioenergy use.")
        if len(severe) == 0:
            actions.append("Until levels are normal, avoid leafy vegetables and root crops, which absorb the most metals.")
        actions.append("Re-test the soil after each season to track progress.")
    else:
        actions.append("No heavy metal contamination above target values. Field can return to normal cropping.")
    plan.append(("Phase 1 - Contamination", actions))

    # Phase 2 - soil health
    actions = []
    if s["ph"] < 6.0:
        actions.append("Apply agricultural lime to raise pH. This also reduces how much metal plants take up.")
    elif s["ph"] > 7.5:
        actions.append("Add compost or manure to slowly bring pH closer to neutral.")
    if s["ec"] > 4:
        actions.append("Salinity is high: leach salts with good-quality water where possible and choose salt-tolerant crops such as barley.")
    if s["n"] < 20:
        actions.append("Nitrogen is low: grow legumes (e.g. vetch, chickpea, lentil) to fix nitrogen naturally.")
    if s["p"] < 10 or s["k"] < 100:
        actions.append("Phosphorus/potassium is low: add compost or manure; apply mineral fertilizer only after expert advice.")
    if not actions:
        actions.append("Soil chemistry is in a good range. Keep adding organic matter to maintain it.")
    plan.append(("Phase 2 - Soil health", actions))

    # Phase 3 - climate resilience
    actions = ["Keep soil covered with cover crops or mulch to protect it from erosion and store carbon."]
    if s["moisture"] < 15:
        actions.append("Soil is dry: choose drought-tolerant crops (e.g. barley, sorghum, chickpea) and use drip irrigation if available.")
    else:
        actions.append("Moisture is adequate: use crop rotation to keep soil healthy and spread risk.")
    actions.append("Reduce tillage where possible to protect soil structure and moisture.")
    plan.append(("Phase 3 - Climate-resilient farming", actions))
    return plan


# ------------------------------------------------------------
# 5) PDF REPORT
# ------------------------------------------------------------
def clean(text):
    """The basic PDF font only supports simple Latin characters (e.g. 'Ţarţūs' -> 'Tartus')."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("‘", "'").replace("’", "'").replace("–", "-")
    return text.encode("latin-1", "replace").decode("latin-1")


def make_pdf(farmer_name, fld, location, safety, metal_df, sensor_rows, plan):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(f"{APP_NAME} - Land Assessment Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, clean("PROTOTYPE - generated from simulated data. Not an official assessment."))
    pdf.ln(3)

    def heading(t):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, clean(t), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

    def line(t):
        pdf.multi_cell(0, 5, clean(t), new_x="LMARGIN", new_y="NEXT")

    heading("1. Field information")
    line(f"Date: {date.today().isoformat()}")
    line(f"Farmer / ID: {farmer_name}")
    line(f"Field: {fld['name']}   |   Size: {fld['size']} ha   |   Previous crop: {fld['crop']}")
    line(f"Location: {location}   |   Coordinates: {fld['lat']:.4f}, {fld['lon']:.4f}")
    pdf.ln(2)

    heading("2. Safety screening")
    line(f"Status: {safety[0]} - {safety[1]}")
    pdf.ln(2)

    heading("3. Heavy metals (mg/kg)")
    for _, r in metal_df.iterrows():
        line(f"- {r['Metal']}: {r['Measured (mg/kg)']} (target {r['Target value']}, "
             f"intervention {r['Intervention value']}) -> {r['Status']}")
    pdf.ln(2)

    heading("4. Soil sensor readings")
    for name, value, finding in sensor_rows:
        line(f"- {name}: {value} -> {finding}")
    pdf.ln(2)

    heading("5. Rehabilitation plan")
    for title, actions in plan:
        pdf.set_font("Helvetica", "B", 10)
        line(title)
        pdf.set_font("Helvetica", "", 10)
        for a in actions:
            line(f"  - {a}")
        pdf.ln(1)

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    line("Reference values: Dutch soil remediation target/intervention values (indicative). "
         "Recommendations are rule-based and must be reviewed by an agronomist.")
    return bytes(pdf.output())


# ------------------------------------------------------------
# 6) FIELD PROFILES (a farmer can have several fields)
# ------------------------------------------------------------
# Every field is stored in st.session_state.fields as a dictionary.
# Widgets are "bound" to these dictionaries so each field keeps its own data
# when you switch between fields.
FIRST_SCENARIO = list(SCENARIOS.keys())[0]


def new_field(name, country=DEFAULT_COUNTRY, place=DEFAULT_PLACE):
    p = PLACE_BY_ID[place]
    sc = SCENARIOS[FIRST_SCENARIO]
    return {
        "name": name, "country": country, "place": place, "coords_for": place,
        "lat": p.lat, "lon": p.lon, "applied_click": None, "size": 2.5, "crop": "Wheat",
        "q1": "No", "q2": "No", "q3": "No", "scenario": FIRST_SCENARIO,
        "metals": {k: float(v) for k, v in sc["metals"].items()},
        "sensors": {k: float(v) for k, v in sc["sensors"].items()},
    }


def wkey(fid, section, name):
    return f"w-{fid}-{section}-{name}"


def forget_widgets(fid, section=None):
    """Remove stored widget values so widgets reload from the field profile."""
    prefix = f"w-{fid}-" if section is None else f"w-{fid}-{section}-"
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[k]


def bind(widget, label, fid, section, name, **kwargs):
    """Show a widget whose value is saved in the field profile."""
    fld = st.session_state.fields[fid]
    store = fld if section == "info" else fld[section]
    k = wkey(fid, section, name)
    if k not in st.session_state:
        st.session_state[k] = store[name]
    st.session_state.bindings[k] = (fid, section, name)
    store[name] = widget(label, key=k, **kwargs)
    return store[name]


def sync_widgets():
    """Copy the latest widget values into the field profiles (runs at the start of every rerun)."""
    for k, (fid, section, name) in list(st.session_state.bindings.items()):
        if fid in st.session_state.fields and k in st.session_state:
            fld = st.session_state.fields[fid]
            store = fld if section == "info" else fld[section]
            store[name] = st.session_state[k]


def add_field():
    current = st.session_state.fields[st.session_state.active]
    fid = st.session_state.next_id
    st.session_state.next_id += 1
    # a new field starts in the same country and place as the current one
    st.session_state.fields[fid] = new_field(f"Field {fid}", current["country"], current["place"])
    st.session_state.active = fid
    st.session_state.active_select = fid


def remove_field():
    fid = st.session_state.active
    del st.session_state.fields[fid]
    forget_widgets(fid)
    st.session_state.active = list(st.session_state.fields)[0]
    st.session_state.active_select = st.session_state.active


def load_scenario(fid):
    fld = st.session_state.fields[fid]
    sc = SCENARIOS[fld["scenario"]]
    fld["metals"] = {k: float(v) for k, v in sc["metals"].items()}
    fld["sensors"] = {k: float(v) for k, v in sc["sensors"].items()}
    forget_widgets(fid, "metals")
    forget_widgets(fid, "sensors")


if "fields" not in st.session_state:
    st.session_state.fields = {1: new_field("Field 1")}
    st.session_state.next_id = 2
    st.session_state.active = 1
    st.session_state.bindings = {}
sync_widgets()


# ------------------------------------------------------------
# 7) USER INTERFACE
# ------------------------------------------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Prototype v0.1 - helping farmers return safely to war-affected land. "
           "Rule-based engine; all soil data shown is simulated.")

with st.sidebar:
    st.header("Farmer")
    farmer_name = st.text_input("Farmer name or ID", "Farmer-001", key="farmer_name")

    st.header("Your fields")
    fields = st.session_state.fields
    if st.session_state.get("active_select") not in fields:
        st.session_state.active_select = st.session_state.active
    st.session_state.active = st.selectbox(
        "Select a field", list(fields), key="active_select",
        format_func=lambda f: fields[f]["name"])
    fid = st.session_state.active
    fld = fields[fid]

    b1, b2 = st.columns(2)
    b1.button("+ Add field", on_click=add_field, width="stretch")
    b2.button("Remove field", on_click=remove_field, disabled=len(fields) == 1, width="stretch")

    st.subheader("Field profile")
    bind(st.text_input, "Field name", fid, "info", "name")

    # --- location: choose from a list so every place is identified correctly ---
    bind(st.selectbox, "Country", fid, "info", "country",
         options=COUNTRY_CODES, format_func=lambda c: COUNTRIES[c])
    place_ids = PLACES[fld["country"]]
    if fld["place"] not in place_ids:          # country changed -> pick its largest place
        fld["place"] = place_ids[0]
        st.session_state.pop(wkey(fid, "info", "place"), None)
    bind(st.selectbox, "Village / town (type to search)", fid, "info", "place",
         options=place_ids, format_func=place_label)
    if fld["coords_for"] != fld["place"]:      # place changed -> move the field pin to that town
        fld["lat"], fld["lon"] = PLACE_BY_ID[fld["place"]].lat, PLACE_BY_ID[fld["place"]].lon
        fld["coords_for"] = fld["place"]
    st.caption("Pick the nearest town, then click your exact field on the map in the "
               "**Field location** tab.")

    bind(st.number_input, "Field size (hectares)", fid, "info", "size", min_value=0.1, max_value=500.0, step=0.5)
    bind(st.text_input, "Crop before the conflict", fid, "info", "crop")

    st.divider()
    bind(st.selectbox, "Example data", fid, "info", "scenario", options=list(SCENARIOS))
    st.button("Load example into this field", on_click=load_scenario, args=(fid,), width="stretch")

location_text = f"{PLACE_BY_ID[fld['place']].name}, {COUNTRIES[fld['country']]}"
st.markdown(f"**Current field:** {fld['name']} - {location_text}")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Field location", "2. Safety check", "3. Soil data", "4. Analysis & plan", "5. Report", "6. Regional map"])

# --- read a new map click FIRST, so everything below already uses the new location ---
# (the key changes with the town, so an old click does not move the pin back)
map_key = f"pick-{fid}-{fld['place']}"
map_state = st.session_state.get(map_key)
click = map_state.get("last_clicked") if isinstance(map_state, dict) else None
if click:
    new = (round(click["lat"], 5), round(click["lng"], 5))
    if new != fld.get("applied_click"):
        fld["lat"], fld["lon"] = new
        fld["applied_click"] = new

# check public hazard data around the field (OpenStreetMap)
hazards, hazard_error = fetch_hazards(fld["lat"], fld["lon"])
nearest_hazard = hazards[0]["distance_km"] if hazards else None

# --- Tab 0: field location (click on the map) ---
with tab0:
    st.subheader("Where exactly is your field?")
    st.write("**Click on your field** on the map to move the blue pin. The map starts at the town you chose. "
             "Switch to **Satellite** (top right) to see the fields themselves.")
    # The base map only depends on the town, so it is NOT reloaded after each click.
    town = PLACE_BY_ID[fld["place"]]
    m = folium.Map(location=[town.lat, town.lon], zoom_start=14, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Map").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
    folium.LayerControl().add_to(m)

    # Things that change (pin + hazard areas) go in a layer that is updated in place.
    fg = folium.FeatureGroup(name="Field")
    for h in hazards:  # mapped hazard areas in red
        for shape in h["shapes"]:
            label = f"{h['kind']} (OpenStreetMap) - {h['distance_km']} km"
            if len(shape) > 2:
                fg.add_child(folium.Polygon(shape, color="#c62828", fill=True, fill_opacity=0.35, tooltip=label))
            else:
                fg.add_child(folium.CircleMarker(shape[0], radius=8, color="#c62828", fill=True, tooltip=label))
    fg.add_child(folium.CircleMarker([fld["lat"], fld["lon"]], radius=11, color="white", weight=3,
                                     fill=True, fill_color="#1565c0", fill_opacity=1,
                                     tooltip=f"Your field: {fld['name']}"))

    st_folium(m, key=map_key, feature_group_to_add=fg, height=480, use_container_width=True,
              returned_objects=["last_clicked"])
    st.caption(f"Field coordinates: {fld['lat']:.5f}, {fld['lon']:.5f}")

# --- Tab 1: safety ---
with tab1:
    st.subheader("Is it safe to work on this field?")
    st.write("Soil sensors are placed **only** on fields that are safe or have been cleared.")

    st.markdown(f"**Public hazard data check** (OpenStreetMap, {HAZARD_RADIUS_M // 1000} km around your field)")
    if hazard_error:
        st.info(hazard_error + " Answer the questions below carefully.")
    elif hazards:
        st.warning(f"{len(hazards)} mapped hazard area(s) found nearby. Nearest: "
                   f"{hazards[0]['kind'].lower()}, {hazards[0]['distance_km']} km away. "
                   "They are shown in red on the Field location map.")
    else:
        st.info("No mapped hazard areas found nearby. This data is incomplete - "
                "it does NOT mean the field is safe.")

    opts = ["No", "Yes", "Not sure"]
    q1 = bind(st.radio, "Is the field listed in a mine-action or hazardous-area record?",
              fid, "info", "q1", options=opts, horizontal=True)
    q2 = bind(st.radio, "Have you or your neighbours seen ordnance, craters or military debris on or near the field?",
              fid, "info", "q2", options=opts, horizontal=True)
    q3 = bind(st.radio, "Has the field been surveyed or cleared by a demining organization?",
              fid, "info", "q3", options=["No", "Yes"], horizontal=True)
    safety = safety_status(q1, q2, q3, nearest_hazard)
    if safety[0] in ("CLEARED", "LOW RISK"):
        st.success(f"**{safety[0]}** - {safety[1]}")
    elif safety[0] == "UNCONFIRMED":
        st.warning(f"**{safety[0]}** - {safety[1]}")
    else:
        st.error(f"**{safety[0]}** - {safety[1]}")

safe_to_continue = safety[0] in ("CLEARED", "LOW RISK")

# --- Tab 2: data input ---
with tab2:
    if not safe_to_continue:
        st.error("Complete the safety check first. Soil testing is only done on safe fields.")
    st.subheader("Baseline test - heavy metals (lab or portable XRF)")
    cols = st.columns(4)
    for i, name in enumerate(METALS):
        with cols[i % 4]:
            bind(st.number_input, f"{name} (mg/kg)", fid, "metals", name, min_value=0.0, max_value=10000.0)
    st.subheader("Soil sensor readings")
    c = st.columns(6)
    with c[0]: bind(st.number_input, "Moisture (%)", fid, "sensors", "moisture", min_value=0.0, max_value=100.0)
    with c[1]: bind(st.number_input, "pH", fid, "sensors", "ph", min_value=0.0, max_value=14.0, step=0.1)
    with c[2]: bind(st.number_input, "EC (dS/m)", fid, "sensors", "ec", min_value=0.0, max_value=50.0, step=0.1)
    with c[3]: bind(st.number_input, "N (mg/kg)", fid, "sensors", "n", min_value=0.0, max_value=1000.0)
    with c[4]: bind(st.number_input, "P (mg/kg)", fid, "sensors", "p", min_value=0.0, max_value=1000.0)
    with c[5]: bind(st.number_input, "K (mg/kg)", fid, "sensors", "k", min_value=0.0, max_value=5000.0)
    st.caption("In the field, these values come from commercial soil sensors. Here they are simulated.")

metal_df = analyse_metals(fld["metals"])
sensor_rows = analyse_sensors(fld["sensors"])
plan = build_plan(metal_df, fld["sensors"])

# --- Tab 3: analysis and plan ---
with tab3:
    if not safe_to_continue:
        st.error("No plan can be generated until the field is confirmed safe.")
    else:
        st.subheader("Heavy metal analysis")
        st.dataframe(metal_df, hide_index=True, width="stretch")
        st.write("**% of intervention value** (100% = serious contamination threshold)")
        st.bar_chart(metal_df.set_index("Metal")["% of intervention value"], horizontal=True)

        st.subheader("Soil health")
        st.dataframe(pd.DataFrame(sensor_rows, columns=["Parameter", "Value", "Finding"]),
                     hide_index=True, width="stretch")

        st.subheader("Your rehabilitation plan")
        for title, actions in plan:
            with st.expander(title, expanded=True):
                for a in actions:
                    if a.startswith("WARNING") or a.startswith("Do NOT"):
                        st.warning(a)
                    else:
                        st.write("- " + a)
        st.caption("Recommendations are rule-based and should be reviewed by an agronomist.")

# --- Tab 4: report ---
with tab4:
    st.subheader(f"Land assessment report - {fld['name']}")
    st.write("A structured report the farmer can attach to applications for rehabilitation programs.")
    if not safe_to_continue:
        st.error("The report is available after the field is confirmed safe.")
    else:
        pdf_bytes = make_pdf(farmer_name, fld, location_text, safety, metal_df, sensor_rows, plan)
        file_name = clean(f"land_report_{farmer_name}_{fld['name']}").replace(" ", "_") + ".pdf"
        st.download_button("Download PDF report", pdf_bytes, file_name=file_name, mime="application/pdf")

# --- Tab 5: regional map ---
with tab5:
    st.subheader("Regional contamination map (anonymized)")
    st.write("When neighbouring farmers use the system, anonymized results build a shared map. "
             "**Your fields** are shown in blue. **Neighbouring points are simulated.**")

    status_rgb = {"Normal": [46, 125, 50], "Elevated": [249, 168, 37], "Seriously contaminated": [198, 40, 40]}

    # simulated neighbouring fields around the current field (same points every time for this place)
    rnd = random.Random(fld["place"])
    neighbours = []
    for _ in range(60):
        status = rnd.choices(list(status_rgb), weights=[6, 3, 1])[0]
        neighbours.append({"lat": fld["lat"] + rnd.uniform(-0.15, 0.15),
                           "lon": fld["lon"] + rnd.uniform(-0.2, 0.2),
                           "rgb": status_rgb[status],
                           "label": f"Neighbouring field (simulated): {status}"})

    # the farmer's own fields (small offset so fields in the same village do not overlap)
    mine = []
    for i, (f_id, f) in enumerate(st.session_state.fields.items()):
        f_status = overall_status(analyse_metals(f["metals"]))
        mine.append({"lat": f["lat"] + 0.004 * i, "lon": f["lon"] + 0.004 * i,
                     "name": f"Your field: {f['name']}",
                     "label": f"Your field: {f['name']} - {f_status}"})

    layers = [
        pdk.Layer("ScatterplotLayer", data=pd.DataFrame(neighbours), get_position=["lon", "lat"],
                  get_fill_color="rgb", get_radius=500, opacity=0.8, pickable=True),
        pdk.Layer("ScatterplotLayer", data=pd.DataFrame(mine), get_position=["lon", "lat"],
                  get_fill_color=[21, 101, 192], get_line_color=[255, 255, 255], stroked=True,
                  line_width_min_pixels=3, get_radius=900, radius_min_pixels=9, pickable=True),
        pdk.Layer("TextLayer", data=pd.DataFrame(mine), get_position=["lon", "lat"], get_text="name",
                  get_size=15, get_color=[13, 71, 161], get_pixel_offset=[0, -24],
                  background=True, get_background_color=[255, 255, 255, 220]),
    ]
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=fld["lat"], longitude=fld["lon"], zoom=10),
        tooltip={"text": "{label}"},
    ))
    st.write("Blue = your fields, Green = normal, Yellow = elevated, Red = seriously contaminated")
