import re


def extract_rx_tx(channel_name):
    rx_match = re.search(r'Rx(\d+)', channel_name)
    tx_match = re.search(r'Tx(\d+)', channel_name)
    return (int(rx_match.group(1)), int(tx_match.group(1))) if rx_match and tx_match else (None, None)


def get_short_map(column_names):
    """
    Returns a mapping of long channels to their corresponding short channels.
    This mapping handles multiple naming conventions and provides detailed diagnostics.
    """
    # First try the manual mapping with exact names
    manual_mapping = {
        'CH0 HbO': 'CH14 HbO',
        'CH1 HbO': 'CH14 HbO',
        'CH2 HbO': 'CH14 HbO',
        'CH3 HbO': 'CH16 HbO',
        'CH4 HbO': 'CH16 HbO',
        'CH5 HbO': 'CH16 HbO',
        'CH6 HbO': 'CH14 HbO',
        'CH7 HbO': 'CH14 HbO',
        'CH8 HbO': 'CH14 HbO',
        'CH9 HbO': 'CH16 HbO',
        'CH10 HbO': 'CH16 HbO',
        'CH11 HbO': 'CH44 HbO',
        'CH12 HbO': 'CH44 HbO',
        'CH13 HbO': 'CH44 HbO',
        'CH15 HbO': 'CH52 HbO',
        'CH17 HbO': 'CH52 HbO',
        'CH18 HbO': 'CH52 HbO',
        'CH19 HbO': 'CH52 HbO',
    }



    # Try exact match first with both naming conventions
    exact_mapping = {long: short for long, short in manual_mapping.items()
                     if long in column_names and short in column_names}


    if exact_mapping:
        print(f"Found {len(exact_mapping)} exact channel mappings")
        return exact_mapping

    # If no exact matches, try to adapt to other naming conventions
    print("No exact channel matches found. Attempting to detect naming pattern...")

    # Try various patterns
    channel_patterns = [
        re.compile(r'(?:CH|Ch|ch)(\d+)[ _]*(O2Hb|HbO|HBO)', re.IGNORECASE),  # CH1 HbO or CH1_HbO format
        re.compile(r'(?:Source|S)(\d+)_(?:Detector|D)(\d+)_(O2Hb|HbO|HBO)', re.IGNORECASE),  # S1_D1_HbO format
        re.compile(r'Tx(\d+)_Rx(\d+)_(O2Hb|HbO|HBO)', re.IGNORECASE),  # Tx1_Rx1_HbO format
    ]

    # Try to match any of the patterns
    matched_cols = {}
    matched_pattern = None

    for pattern in channel_patterns:
        possible_matches = {}
        for col in column_names:
            match = pattern.search(col)
            if match:
                if len(match.groups()) == 2:  # CH# HbO format
                    ch_num = int(match.group(1))
                    possible_matches[ch_num] = col
                elif len(match.groups()) == 3:  # Source/Detector or Tx/Rx format
                    # For simplicity, we'll just use the first number as an index
                    ch_num = int(match.group(1))
                    possible_matches[ch_num] = col

        if possible_matches:
            matched_cols = possible_matches
            matched_pattern = pattern
            break

    if not matched_cols:
        print("WARNING: Could not identify any channel pattern in column names")
        print(f"Column examples: {column_names[:5]}")
        return {}

    print(f"Detected channel pattern: {matched_pattern.pattern}")
    print(f"Found {len(matched_cols)} channels matching this pattern")

    # Get the actual short channel numbers
    short_channel_numbers = [14, 16, 44, 52]
    short_channels = {ch_num: matched_cols[ch_num] for ch_num in short_channel_numbers
                      if ch_num in matched_cols}

    if not short_channels:
        print("WARNING: None of the expected short channels (14, 16, 44, 52) were found!")
        return {}

    print(f"Found {len(short_channels)} short channels: {list(short_channels.keys())}")

    # Create mapping based on the detected channels
    short_map_refs = {
        0: 14, 1: 14, 2: 14, 3: 16, 4: 16, 5: 16,
        6: 14, 7: 14, 8: 14, 9: 16, 10: 16, 11: 44,
        12: 44, 13: 44, 15: 52, 17: 52, 18: 52, 19: 52
    }

    # Create a new mapping based on the detected naming convention
    adapted_mapping = {}
    for long_num, short_num in short_map_refs.items():
        if long_num in matched_cols and short_num in short_channels:
            adapted_mapping[matched_cols[long_num]] = short_channels[short_num]

    print(f"Created {len(adapted_mapping)} channel mappings")
    return adapted_mapping


# Region definitions using CH# format
regions = {
    "PFC_L": [f"CH{ch} HbO" for ch in [4, 6, 7]] + [f"CH{ch} HHb" for ch in [4, 6, 7]],
    "PFC_R": [f"CH{ch} HbO" for ch in [0, 1, 2]] + [f"CH{ch} HHb" for ch in [0, 1, 2]],
    "SMA_L": [f"CH{ch} HbO" for ch in [11, 12]] + [f"CH{ch} HHb" for ch in [11, 12]],
    "SMA_R": [f"CH{ch} HbO" for ch in [15, 16]] + [f"CH{ch} HHb" for ch in [15, 16]],
    "M1_L":  [f"CH{ch} HbO" for ch in [9, 13]] + [f"CH{ch} HHb" for ch in [9, 13]],
    "M1_R":  [f"CH{ch} HbO" for ch in [17, 19]] + [f"CH{ch} HHb" for ch in [17, 19]],
    "S1_L":  [f"CH{ch} HbO" for ch in [10, 14]] + [f"CH{ch} HHb" for ch in [10, 14]],
    "S1_R":  [f"CH{ch} HbO" for ch in [18, 20]] + [f"CH{ch} HHb" for ch in [18, 20]],
    "V1_L":  [f"CH{ch} HbO" for ch in [8]]      + [f"CH{ch} HHb" for ch in [8]],
    "V1_R":  [f"CH{ch} HbO" for ch in [21]]     + [f"CH{ch} HHb" for ch in [21]],
}



def adjust_regions_to_naming(column_names):
    """
    Since we're using consistent HbO/HHb naming, just verify columns exist
    """
    valid_regions = {}
    for region, channels in regions.items():
        valid_channels = [ch for ch in channels if ch in column_names]
        if valid_channels:
            valid_regions[region] = valid_channels
    return valid_regions


# Add HHb counterparts automatically
for region in list(regions.keys()):
    hhbs = [ch.replace("HbO", "HHb") for ch in regions[region]]
    regions[region].extend(hhbs)