import pandas as pd
import numpy as np

def df_convoys(convoys):
    convoy_data = []

    # Loop through each convoy to collect its information
    for convoy in convoys:
        # Append a dictionary for each convoy with its start time, end time, and atoms
        convoy_data.append({
            'Start': convoy.start_time,
            'End': convoy.end_time,
            'HB Atoms': convoy.atoms
        })

    # Convert the list of dictionaries into a pandas DataFrame
    convoy_df = pd.DataFrame(convoy_data).sort_values(by='Start').reset_index(drop=True)
    total = convoy_df['End'] - convoy_df['Start'] + 1
    print(f"Total: {total.sum()}")

    # Display the DataFrame
    return convoy_df

def BARMCNaive(results, t1, t2, k):

    # Initialize an empty DataFrame to store relaxed blocks
    grouped = results.groupby(['H', 'N', 'O'])

    combined_ranges = []

    for _, group in grouped:
        group = group.sort_values(by='start')
        combined_start = group.iloc[0]['start']
        combined_end = group.iloc[0]['end']
        total_gaps = 0

        for i in range(1, len(group)):
            gap = group.iloc[i]['start'] - combined_end - 1
            if gap <= t1:
                combined_end = group.iloc[i]['end']
                total_gaps += gap
            else:
                sequence_length = combined_end - combined_start + 1
                if sequence_length > k and (total_gaps / sequence_length) <= t2:
                    combined_ranges.append((combined_start, combined_end, (group.iloc[0]['H'], group.iloc[0]['N'], group.iloc[0]['O'])))
                combined_start = group.iloc[i]['start']
                combined_end = group.iloc[i]['end']
                total_gaps = 0

        # Check the last combined range after exiting the loop
        sequence_length = combined_end - combined_start + 1
        if sequence_length > k and (total_gaps / sequence_length) <= t2:
            combined_ranges.append((combined_start, combined_end, (group.iloc[0]['H'], group.iloc[0]['N'], group.iloc[0]['O'])))

    # Convert the combined ranges to a DataFrame
    df_combined_ranges = pd.DataFrame(combined_ranges, columns=['start', 'end', 'Indices'])
    return df_combined_ranges

def find_consecutive_instances(data, k):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data, columns=['Time', 'Condition', 'H', 'N', 'O'])

    # df = pd.DataFrame(data)
    
    # Ensure data types are correct
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Condition'] = df['Condition'].astype(bool)
    df['H'] = pd.to_numeric(df['H'], errors='coerce')
    df['N'] = pd.to_numeric(df['N'], errors='coerce')
    df['O'] = pd.to_numeric(df['O'], errors='coerce')

    # Sort data first by H, N, O and then by Time
    df.sort_values(by=['H', 'N', 'O', 'Time'], inplace=True)

    # Identify consecutive rows where Condition is True and Time is consecutive
    df['group'] = ((df['H'] != df['H'].shift()) |
                   (df['N'] != df['N'].shift()) |
                   (df['O'] != df['O'].shift()) |
                   (df['Condition'] == False) |
                   (df['Time'] - df['Time'].shift() != 1)).cumsum()

    # display(df)
    # Filter out rows where Condition is False
    df = df[df['Condition']]

    # Group by the new 'group' identifier and get the first and last time
    grouped = df.groupby('group')
    result = grouped['Time'].agg(start='min', end='max')
    result[['H', 'N', 'O']] = grouped[['H', 'N', 'O']].first()

    # Adding a unique identifier column for each block (BAMC Number)
    result.reset_index(drop=True, inplace=True)
    result.index.name = 'BAMC'
    result.reset_index(inplace=True)

    # Reordering columns for clarity
    result = result[['BAMC', 'start', 'end', 'H', 'N', 'O']]
    results = result[(result['end'] - result['start'] + 1) >= k]

    return results

def hydrogenBondCheckGivenIndices(start, end, indices, data, atomType):
    atomsIHave = hb_atom_types_in_frame(indices, atomType)

    # print(atomsIHave)

    # hb = False
    arr = np.empty(shape=(0, 5))
    for timeFrame in range(start, end+1):
        # print(data[timeFrame].shape)
        HBAtoms = find_hydrogen_bonds(data[timeFrame], atomsIHave)
        if HBAtoms == []:
            arr = np.vstack((arr, np.array([[timeFrame, False, None, None, None]])))
        else:
            # print(timeFrame, HBAtoms)
            for HBAtom in HBAtoms:
                arr = np.vstack((arr, np.array([[timeFrame, True, HBAtom[0], HBAtom[1], HBAtom[2]]])))
    return arr

def calculateAtomTypes(fileLocation):
    df = pd.read_csv(fileLocation)

    # get indices of atoms with atom_type 'n' and subst_id between 1-4
    n_atoms = df[(df['atom_type'] == 'n') & (df['subst_id'] >= 1) & (df['subst_id'] <= 4)].index.tolist()
    hn_atoms = df[(df['atom_type'] == 'hn') & (df['subst_id'] >= 1) & (df['subst_id'] <= 4)].index.tolist()

    # get indices of atoms with atom_type 'o' or 'os' and subst_id between 5-14
    o_atoms = df[((df['atom_type'] == 'o') | (df['atom_type'] == 'os')) & (df['subst_id'] >= 5) & (df['subst_id'] <= 14)].index.tolist()

    atomType = {
        "n": n_atoms,
        "hn": hn_atoms,
        "o": o_atoms
    }

    return atomType

def hb_atom_types_in_frame(indices, atomType):
    atomsIHave = {
            "n": [],
            "hn": [],
            "o": []
        }

    for index in indices:
        if not find_key_by_value(index, atomType) == None:
            atomsIHave[find_key_by_value(index, atomType)].append(index)

    return atomsIHave

def compute_angles(vectors_hn_n, vectors_hn_o):
    """
    Computes angles between HN-N and HN-O vectors.
    """
    # Normalize vectors
    norm_hn_n = np.linalg.norm(vectors_hn_n, axis=1, keepdims=True)
    norm_hn_o = np.linalg.norm(vectors_hn_o, axis=1, keepdims=True)
    unit_hn_n = vectors_hn_n / norm_hn_n
    unit_hn_o = vectors_hn_o / norm_hn_o

    # Dot product and angle
    dot_product = np.sum(unit_hn_n * unit_hn_o, axis=1)
    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return angles

def find_hydrogen_bonds(data, atomsIHave):
    hn_indices = np.array(atomsIHave['hn'])
    n_indices = np.array(atomsIHave['n'])
    o_indices = np.array(atomsIHave['o'])

    if len(hn_indices) == 0 or len(n_indices) == 0 or len(o_indices) == 0:
        return []

    # Compute distances between N and O, and get displacement vectors
    distances_n_o, vectors_n_o = get_vectorized_distances(data, n_indices, o_indices)
    n_o_candidate_mask = distances_n_o <= 3.5
    
    valid_hbonds = []
    
    # Iterate through N-O pairs that meet the distance criteria
    for n_idx, o_idx in zip(*np.where(n_o_candidate_mask)):
        n_coord = data[n_indices[n_idx]]
        o_coord = data[o_indices[o_idx]]

        # For each N-O pair, find the corresponding H atoms
        for hn_idx in hn_indices:
            hn_coord = data[hn_idx]

            # Compute vectors
            vector_hn_n = n_coord - hn_coord
            vector_hn_o = o_coord - hn_coord

            # Calculate angle
            angle = compute_angles(vector_hn_n[np.newaxis, :], vector_hn_o[np.newaxis, :])[0]

            # Check angle criteria
            if 2.35619 < angle <= 3.14159:
                valid_hbonds.append([hn_idx, n_indices[n_idx], o_indices[o_idx]])

    return valid_hbonds

def find_key_by_value(input_value, dictionary):
    for key, value in dictionary.items():
        if input_value in value:
            return key
    return None

def get_vectorized_distances(data, indices_a, indices_b):
    """
    Computes vectorized distances between two sets of atoms, considering periodic boundaries.
    """
    coords_a = data[indices_a][:, np.newaxis, :]  # Shape: (len(indices_a), 1, 3)
    coords_b = data[indices_b]                    # Shape: (len(indices_b), 3)
    diff = coords_a - coords_b
    box_size = 72.475
    diff -= box_size * np.round(diff / box_size)  # Apply periodic boundary conditions
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances, diff

def filterDF(df):
    osDF = df[((df['atom_type'] == 'o') | (df['atom_type'] == 'os')) & (df['subst_id'] >= 5) & (df['subst_id'] <= 14)]
    nDF = df[(df['atom_type'] == 'n') & (df['subst_id'] >= 1) & (df['subst_id'] <= 4)]
    o_and_n = pd.concat([osDF, nDF])
    # display(o_and_n)
    o_and_n = o_and_n.reset_index()
    o_and_n_indices = {}
    for key, val in o_and_n.groupby('subst_id').indices.items():
        o_and_n_indices[key] = o_and_n.loc[val]["index"].values

    return o_and_n_indices

def hb_atom_types_in_data(allInfo):
    n_atoms = allInfo[(allInfo['atom_type'] == 'n') & (allInfo['subst_id'] >= 1) & (allInfo['subst_id'] <= 4)].index.tolist()
    hn_atoms = allInfo[(allInfo['atom_type'] == 'hn') & (allInfo['subst_id'] >= 1) & (allInfo['subst_id'] <= 4)].index.tolist()

    # get indices of atoms with atom_type 'o' or 'os' and subst_id between 5-14
    o_atoms = allInfo[((allInfo['atom_type'] == 'o') | (allInfo['atom_type'] == 'os')) & (allInfo['subst_id'] >= 5) & (allInfo['subst_id'] <= 14)].index.tolist()

    atomType = {
        "n": n_atoms,
        "hn": hn_atoms,
        "o": o_atoms
    }

    return atomType


def extract_semi_vectorized(data, nitrogens_and_oxygens_indices, threshold):
    # Existing setup remains unchanged
    coords_i = np.vstack([data[nitrogens_and_oxygens_indices[i]] for i in range(1, 5)])
    coords_j = np.vstack([data[nitrogens_and_oxygens_indices[j]] for j in range(5, 15)])
    
    offsets_i = np.cumsum([0] + [len(nitrogens_and_oxygens_indices[i]) for i in range(1, 5)])
    offsets_j = np.cumsum([0] + [len(nitrogens_and_oxygens_indices[j]) for j in range(5, 15)])
    
    dists = np.sqrt(np.sum((coords_i[:, np.newaxis, :] - coords_j[np.newaxis, :, :]) ** 2, axis=-1))
    
    idx_i, idx_j = np.where(dists < threshold)

    # Vectorize mapping back to original groups
    # Create an array where each element's value is its group number
    group_map_i = np.zeros(len(coords_i), dtype=int)
    for i, offset in enumerate(offsets_i[:-1], start=1):
        group_map_i[offsets_i[i-1]:offsets_i[i]] = i
    
    group_map_j = np.zeros(len(coords_j), dtype=int)
    for j, offset in enumerate(offsets_j[:-1], start=5):
        group_map_j[offsets_j[j-5]:offsets_j[j-5+1]] = j
    
    # Use the group_map arrays to find the original group for each index
    orig_i = group_map_i[idx_i]
    orig_j = group_map_j[idx_j]

    # Calculate relative indices within each group
    ii_rel = idx_i - np.searchsorted(offsets_i, idx_i, side='right') + 1
    jj_rel = idx_j - np.searchsorted(offsets_j, idx_j, side='right') + 5

    # Compile results (vectorized)
    results = np.vstack((orig_i, orig_j, ii_rel, jj_rel)).T

    return results.tolist()