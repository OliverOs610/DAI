import numpy as np
import pandas as pd
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import os
import gc
import time
from numba import jit
from pathlib import Path

# ==========================================
# 1. KONSTANTEN & HILFSFUNKTIONEN
# ==========================================
# Feiertage als frozenset für schnelleren Lookup
@jit(nopython=True)
def get_china_daytype_numba(years, months, days):
    """Numba-optimierte Daytype-Berechnung mit korrigierter Variablendefinition."""
    # Länge der Eingabe (wichtig für Numba!)
    n = len(years)
    daytype = np.ones(n, dtype=np.int8)

    # Feiertage als lokale Liste (Numba-kompatibel)
    holidays = [
        # 2010
        20100101, 20100102, 20100103, 20100213, 20100214, 20100215, 20100216, 20100217, 20100218, 20100219,
        20100403, 20100404, 20100405, 20100501, 20100502, 20100503, 20100614, 20100615, 20100616, 20100922,
        20100923, 20100924, 20101001, 20101002, 20101003, 20101004, 20101005, 20101006, 20101007,
        # 2011
        20110101, 20110102, 20110103, 20110202, 20110203, 20110204, 20110205, 20110206, 20110207,
        20110208, 20110403, 20110404, 20110405, 20110501, 20110502, 20110503, 20110604, 20110605,
        20110606, 20110910, 20110911, 20110912, 20111001, 20111002, 20111003, 20111004, 20111005,
        20111006, 20111007,
        # 2012
        20120101, 20120102, 20120103, 20120122, 20120123, 20120124, 20120125, 20120126, 20120127,
        20120128, 20120402, 20120403, 20120404, 20120429, 20120430, 20120501, 20120622, 20120623,
        20120624, 20120930, 20121001, 20121002, 20121003, 20121004, 20121005, 20121006, 20121007,
        # 2013
        20130101, 20130102, 20130103, 20130209, 20130210, 20130211, 20130212, 20130213, 20130214,
        20130215, 20130404, 20130405, 20130406, 20130429, 20130430, 20130501, 20130610, 20130611,
        20130612, 20130919, 20130920, 20130921, 20131001, 20131002, 20131003, 20131004, 20131005,
        20131006, 20131007,
        # 2014
        20140101, 20140131, 20140201, 20140202, 20140203, 20140204, 20140205, 20140206, 20140405,
        20140406, 20140407, 20140501, 20140502, 20140503, 20140531, 20140601, 20140602, 20140906,
        20140907, 20140908, 20141001, 20141002, 20141003, 20141004, 20141005, 20141006, 20141007,
        # 2015
        20150101, 20150102, 20150103, 20150218, 20150219, 20150220, 20150221, 20150222, 20150223,
        20150224, 20150404, 20150405, 20150406, 20150501, 20150502, 20150503, 20150620, 20150621,
        20150622, 20150926, 20150927, 20151001, 20151002, 20151003, 20151004, 20151005, 20151006,
        20151007
    ]

    for i in range(n):
        year, month, day = years[i], months[i], days[i]

        # Schaltjahrprüfung (Numba-kompatibel)
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        month_days = [31, 28 + is_leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Wochentag berechnen (Zeller's Congruence)
        if month < 3:
            month += 12
            year -= 1
        K = year % 100
        J = year // 100
        weekday = (day + (13 * (month + 1)) // 5 + K + (K // 4) + (J // 4) + 5 * J) % 7

        # Daytype-Logik
        if weekday == 5:  # Samstag
            daytype[i] = 2
        elif weekday == 6:  # Sonntag
            daytype[i] = 3
        else:
            # Feiertagsprüfung (lineare Suche)
            date_hash = year * 10000 + month * 100 + day
            for holiday in holidays:
                if date_hash == holiday:
                    daytype[i] = 3
                    break
    return daytype

# ==========================================
# 2. DATEN LADEN (NUR Beijing)
# ==========================================

skript_verzeichnis = Path(__file__).parent
edgardaten = skript_verzeichnis / 'MiniAllPolutantsMonthly.parquet'
try:
    df_edgar = pd.read_parquet(
        edgardaten,
        filters=[('city', '==', 'Beijing'), ('year', '==', 2010),('month', '==', 1)]
    )
    print(f"✅ Beijing-Daten geladen: {len(df_edgar)} Zeilen")
except Exception as e:
    raise SystemExit(f"❌ Fehler beim Laden der EDGAR-Daten: {e}")

profildaten = skript_verzeichnis / 'hourly_profiles_china2.csv'
try:
    df_profiles = pd.read_csv(profildaten)
    # Vorab Gewichte pro (month_id, Daytype_id) berechnen; 30 kommt daher, dass es 30 "acivity_code"s gibt 
    weight_sums = (
        df_profiles
        .groupby(['month_id', 'Daytype_id'])[
            [f'h{i}' for i in range(1, 25)]
        ]
        .sum()
        .sum(axis=1)
    )
    print("✅ Profil-Daten erfolgreich geladen")
except Exception as e:
    raise SystemExit(f"❌ Fehler beim Laden der Profile: {e}")

# ==========================================
# 3. CHUNKING-VERARBEITUNG (OPTIMIERT)
# ==========================================
temp_dir = "ChunkBeij"
os.makedirs(temp_dir, exist_ok=True)

chunk_size = 2000
num_chunks = (len(df_edgar) - 1) // chunk_size + 1
chunk_files = []

# 1. Eindeutige activity_codes identifizieren
unique_activities = df_profiles['activity_code'].unique()
print(f"Gefundene Activity-Codes: {unique_activities}")

# 2. Mapping von activity_code zu Index erstellen
activity_to_idx = {code: idx for idx, code in enumerate(unique_activities)}
n_activities = len(unique_activities)
# 3. 4D-Array erstellen: [activity, month, daytype, hour]
weight_array_4d = np.zeros((n_activities, 12, 3, 24), dtype=np.float64)  # activity x month × daytype × hour
# 4. Daten füllen
for _, row in df_profiles.iterrows():
    # Indizes für die ersten 3 Dimensionen
    activity_idx = activity_to_idx[row['activity_code']]
    month_idx = row['month_id'] - 1      # 0-basiert
    daytype_idx = row['Daytype_id'] - 1  # 0-basiert
    
    # Stunden-Werte (h1 bis h24) in das Array schreiben
    for h in range(1, 25):
        weight_array_4d[activity_idx, month_idx, daytype_idx, h-1] = row[f'h{h}']
        
for i in range(0, len(df_edgar), chunk_size):
    chunk_num = i // chunk_size + 1
    chunk = df_edgar.iloc[i:i+chunk_size].copy()
    print(f"Verarbeite Chunk {chunk_num}/{num_chunks}")

    # 1. Tage pro Monat (vektorisiert)
    start = time.time()
    chunk['days_in_month'] = pd.to_datetime(
        chunk['year'] * 100 + chunk['month'], format='%Y%m'
    ).dt.days_in_month
    #print(f"Tage pro Monat: {time.time() - start:.2f} Sekunden")
    # 2. Tage expandieren (NumPy-optimiert)
    indices = np.repeat(np.arange(len(chunk)), chunk['days_in_month'])
    days = np.concatenate([np.arange(1, d+1) for d in chunk['days_in_month']])
    chunk_expanded = pd.DataFrame({
        **{col: chunk[col].values[indices] for col in chunk.columns},
        'day': days
    })
    #print(f"Tage-Expansion: {time.time() - start:.2f} Sekunden")
    # 3. Daytype (Numba-optimiert)
    chunk_expanded['daytype'] = get_china_daytype_numba(
    chunk_expanded['year'].values,
    chunk_expanded['month'].values,
    chunk_expanded['day'].values
)
    #print(f"Daytype: {time.time() - start:.2f} Sekunden")
# 4. Stunden-Expansion (korrigierte 3D-Array-Version)
    start = time.time()

# Erzeuge alle Stunden (1-24) für jede Zeile in chunk_expanded
    hours = np.tile(np.arange(1, 25), len(chunk_expanded))

# Wiederhole month/daytype für jede Stunde
    month_expanded = np.repeat(chunk_expanded['month'].values, 24)
    daytype_expanded = np.repeat(chunk_expanded['daytype'].values, 24)
    #idx = activity_to_idx[activity_name]
# Hole Gewichte direkt aus dem 3D-Array (keine Schleifen!)
    weights = weight_array_4d[
        idx,
        month_expanded - 1,  # month_id → 0-basiert
        daytype_expanded - 1,  # Daytype_id → 0-basiert
        hours - 1  # hour → 0-basiert
    ]

    #print(f"Stunden-Expansion: {time.time() - start:.2f} Sekunden")

# 5. Emissionen berechnen (korrigierte Version)
    emissions_daily_kg = (
    chunk_expanded['emissions'].values 
    / chunk_expanded['days_in_month'].values  # Emissionen pro Tag
    * 1000  # Mg → kg
    )

# Auf Stunden expandieren
    emissions_daily_kg_exp = np.repeat(emissions_daily_kg, 24)

# Korrigierter MultiIndex-Lookup
    weight_sums_expanded = weight_sums[
        list(zip(chunk_expanded['month'], chunk_expanded['daytype']))
    ].values.repeat(24)

    emissions_hourly = emissions_daily_kg_exp * weights / weight_sums_expanded
# 6. Finaler DataFrame (mit Fehlerbehandlung)
    df_hourly = pd.DataFrame({
    'city': 'Beijing',
    'datetime': pd.to_datetime({
        'year': chunk_expanded['year'].values.repeat(24),
        'month': chunk_expanded['month'].values.repeat(24),
        'day': chunk_expanded['day'].values.repeat(24),
        'hour': hours
    }),
    'pollutant': chunk_expanded['pollutant'].values.repeat(24),
    'lat': chunk_expanded['lat'].values.repeat(24),
    'lon': chunk_expanded['lon'].values.repeat(24),
    'sector': chunk_expanded['sector'].values.repeat(24),
    'activity_code': np.repeat('DEFAULT', len(chunk_expanded) * 24),  # Immer string
    'emissions_kg': emissions_hourly
})
    # 7. Chunk speichern (PyArrow-optimiert)
    chunk_file = os.path.join(temp_dir, f"chunk_{chunk_num:04d}.parquet")
    table = pa.Table.from_pandas(df_hourly, preserve_index=False)
    pq.write_table(table, chunk_file, compression='snappy')
    chunk_files.append(chunk_file)
    print(f"  → Chunk {chunk_num} gespeichert: {chunk_file}")
    print(f"Chunk speichern: {time.time() - start:.2f} Sekunden")

    # 8. Speicher bereinigen
    del chunk, chunk_expanded, df_hourly,
    del emissions_daily_kg, emissions_daily_kg_exp, emissions_hourly, weights, weight_sums_expanded
    gc.collect()
    print(f"Speicher-bereinigen: {time.time() - start:.2f} Sekunden")
    
# ==========================================
# 4. CHUNKS ZUSAMMENFÜHREN
# ==========================================
print(f"\n📊 Führe {len(chunk_files)} Chunks zusammen...")

schema = None
writer = None

for i, f in enumerate(chunk_files):
    table = pq.read_table(f)
    if schema is None:
        schema = table.schema
        writer = pq.ParquetWriter("MiniBeijing.parquet", schema, compression='snappy')
    else:
        # Schema anpassen, falls nötig
        if not table.schema.equals(schema):
            table = table.cast(schema)
    writer.write_table(table)
    del table
    gc.collect()

if writer:
    writer.close()

print("💾 Ergebnisse gespeichert: Expl_few_beijing.parquet")

# Temporäre Files löschen
print(f"\n🗑️  Lösche {len(chunk_files)} temporäre Chunk-Files...")
for f in chunk_files:
    os.remove(f)
os.rmdir(temp_dir)
print("✅ Verarbeitung abgeschlossen!")