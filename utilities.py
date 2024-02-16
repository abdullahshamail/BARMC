from hydrogenbond import HydrogenBond
import numpy as np
import pandas as pd

def printConvoys(convoys):
    print("Total number of Convoys =", len(convoys))
    for convoy in convoys:
        print('No of elements in the Convoy', len(convoy.indices))  
        print(convoy.start_time, convoy.end_time, convoy.indices)

def getAtomType(index):
    atomTypes = {
                "hn": [0, 216], 
                 "n": [216, 432], 
                 "o": [432, 442], 
                 "os": [442, 452],
                 }
    for atom, range in atomTypes.items():
        if index >= range[0] and index < range[1]:
            return atom
    
    return "none"

def find_key_by_value(input_value, dictionary):
    for key, value in dictionary.items():
        if input_value in value:
            return key
    return None



def extractInterestingMoleculesPerTimeframe(data, nitrogens_and_oxygens_indices, threshold):
    results = []  # list to store results
    for i in range(1, 5):
        # iterate over second set of subst_id groups
        for j in range(5, 15):
            coords_i = data[nitrogens_and_oxygens_indices[i]]
            coords_j = data[nitrogens_and_oxygens_indices[j]]
            # compute distances between atoms in the two subst_id groups
            dists = np.linalg.norm(coords_i[:, np.newaxis, :] - coords_j, axis=2)
            # find indices of atoms that are within x units of each other
            idx_i, idx_j = np.where(dists < threshold)
            # append results to list
            for ii, jj in zip(idx_i, idx_j):
                results.append((i, j, ii ,jj))

    # print results
    return results


def split_xyz (coords):
    """
    Parameters
    ----------
    coords : 3D Coordinates

    Returns
    -------
    Coordinates on each axis
 
    """
    # Calculate the x coordinates
    X1 = coords[:,0]

    # Calculate the y coordinates
    Y1 = coords[:,1]

    # Calculate the z coordinates
    Z1 = coords[:,2]
    
    return X1, Y1, Z1


def is_different_atom_set(current, previous):
    return any(current[1:] != previous[1:])

def BAConvoys(data, k):
    ranges = []

    # Iterate through the array
    for i, row in enumerate(data):
        # Check if the row has HB True
        if row[0] == True:
            # Check for the first row or a different atom set
            if i == 0 or is_different_atom_set(row, data[i-1]):
                start = i
            # Check for the last row or a different atom set in the next row
            if i == len(data) - 1 or (data[i+1][0] == False or is_different_atom_set(row, data[i+1])):
                end = i
                if (end - start) >= k:
                    ranges.append((start, end, row[1:]))

    # Convert the ranges to a DataFrame
    df_ranges = pd.DataFrame(ranges, columns=['Start', 'End', 'Indices'])

    return df_ranges


def consecutive_true_indices(bool_list, k):
    true_indices = [i for i, x in enumerate(bool_list) if x]
    consecutive_ranges = []
    current_range = []
    for i in range(len(true_indices)):
        if not current_range:
            current_range.append(true_indices[i])
        elif true_indices[i] == current_range[-1] + 1:
            current_range.append(true_indices[i])
        else:
            if len(current_range) >= k:
                consecutive_ranges.append(current_range)
            current_range = [true_indices[i]]
    if current_range and len(current_range) >= k:
        consecutive_ranges.append(current_range)
    output_str = ''
    for r in consecutive_ranges:
        if len(r) == 1:
            output_str += str(r[0]) + ', '
        else:
            output_str += str(r[0]) + '-' + str(r[-1]) + ', '
    return output_str[:-2]

def combine_ranges(ranges, t):
    combined_ranges = []
    start, end = ranges[0]
    for i in range(1, len(ranges)):
        next_start, next_end = ranges[i]
        if next_start - end <= t:
            end = next_end
        else:
            combined_ranges.append((start, end))
            start, end = next_start, next_end
    combined_ranges.append((start, end))
    return combined_ranges


def hydrogenBondCheckGivenIndices(start, end, indices, data, atomType):
    atomsIHave = {
                    "n": [],
                    "hn": [],
                    "o": []
                }
    for index in indices:
        if not find_key_by_value(index, atomType) == None:
            atomsIHave[find_key_by_value(index, atomType)].append(index)

    hb = False
    arr = np.empty(shape=(0, 4))
    for timeFrame in range(start, end+1):

        for hn in atomsIHave["hn"]:
            for n in atomsIHave["n"]:
                coord_hn = getCoordinates(data[timeFrame], hn)
                coord_n = getCoordinates(data[timeFrame], n)
                for o in atomsIHave["o"]:
                    coord_o = getCoordinates(data[timeFrame], o)
                    if HB_truth(coord_hn, coord_n, coord_o):
                        hb = True

                        break
                if hb: break
            if hb: break

        if hb:
            arr = np.vstack((arr, np.array([[hb, hn, n, o]])))
        else:
            arr = np.vstack((arr, np.array([[hb, None, None, None]])))
        hb = False
    return arr


def hydrogenBondCheckGivenIndices2(start, end, indices, data, atomType, timeGap):
    atomsIHave = {
                    "n": [],
                    "hn": [],
                    "o": []
                }
    for index in indices:
        if not find_key_by_value(index, atomType) == None:
            atomsIHave[find_key_by_value(index, atomType)].append(index)

    tempGap = 0
    hb = False
    for timeFrame in range(start, end):
        for hn in atomsIHave["hn"]:
                for n in atomsIHave["n"]:
                    coord_hn = getCoordinates(data[timeFrame], hn)
                    coord_n = getCoordinates(data[timeFrame], n)
                    for o in atomsIHave["o"]:
                        coord_o = getCoordinates(data[timeFrame], o)
                        if HB_truth(coord_hn, coord_n, coord_o):
                            # print(start, end, timeFrame, hn, n, o)
                            # return True
                            hb = True
        if tempGap == timeGap  and timeGap != 0: return False
        tempGap = 0 if hb else tempGap + 1
    return True if hb else False


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


def getAtomTypesFromArray(indices):
    returnDict = {}
    atomTypes = {
                "hn": [0, 216], 
                 "n": [216, 432], 
                 "o": [432, 442], 
                 "os": [442, 452],
                 }
    for atom, range in atomTypes.items():
        returnDict[atom] = indices[(indices >= range[0]) & (indices < range[1])]
    return returnDict

def getCoordinates(data, index):
    return data[index]

def coordinate_transform (x):
    x = 72.475 + x  if x < 0 else x
    return x

def periodic_boundary (x1, x2):

    """
    Apply periodic boundaries
    Periodic Boundaries in X and Y direction is 72 A.
    Z axis does not have any periodic boundary
    
    So, the periodicity conditions

    Difference in X axis = if |x2 − x1|> 36:
                                72 −|x2 − x1|
                            
                         else:
                                x2 − x1
    
    Difference in Y axis = same as X

    Parameters
    ----------
    x1 : axis coordinates of atom 1  
    x2 : axis coordinates of atom 2 

    Returns
    -------
    differences of coordinates values with periodicity

    """
    
    x1 = coordinate_transform (x1)
    
    x2 = coordinate_transform (x2)

    coord_diff = ((x2 - x1)/abs(x2 - x1))*(72.475-abs(x2 - x1)) if abs(x2 - x1)>36.2375 else (x2 - x1)
    
    return coord_diff

def distance_periodicity (atom1, atom2):
    
    """
    Apply periodic boundaries
    
    Periodic Boundaries in X and Y direction is 72 A.
    Z axis does not have any periodic boundary
    
    So, the periodicity conditions

    Distance in X axis = if |x2 − x1|> 36:
                                72 −|x2 − x1|
                            
                         else:
                                x2 − x1
    
    Distance in Y axis = same as X
    
    Euclidean Shortest Distance = √(〖"(Distance in X axis)" 〗^2+〖"(Distance in Y axis)" 〗^2+〖"(Distance in Z axis)" 〗^2 )

    Parameters
    ----------
    X1 : Numpy 1D array
        x axis coordinates of atom 1 trajectory
    Y1 : Numpy 1D array
        y axis coordinates of atom 1 trajectory 
    Z1 : Numpy 1D array
        z axis coordinates of atom 1 trajectory 
    X2 : Numpy 1D array
        x axis coordinates of atom 2 trajectory 
    Y2 : Numpy 1D array
        y axis coordinates of atom 2 trajectory 
    Z2 : Numpy 1D array
        z axis coordinates of atom 2 trajectory 

    Returns
    -------
    distance: Numpy 1D array
            Euclidean distance of (X1,Y1,Z1) and (X2,Y2,Z2)

    """

    x_coord_diff = periodic_boundary (atom1[0], atom2[0])
        
    y_coord_diff = periodic_boundary (atom1[1], atom2[1])

    z_coord_diff = (atom2[2]-atom1[2])

    euclidean_distance = np.sqrt(np.square(x_coord_diff) + np.square(y_coord_diff) + np.square(z_coord_diff))
    
    return euclidean_distance, x_coord_diff, y_coord_diff, z_coord_diff


def calc_angle (vector1,vector2):
    
    """ Returns the angle between two vectors  """ 
    
    c = np.dot(vector1,vector2) / (np.linalg.norm(vector1)* np.linalg.norm(vector2))
#     angle = np.arccos(np.clip(c, -1 , 1))
    angle = np.arccos(c)
    return angle


def HB_truth(hn, n, o):
    euclidean_distance_no, x_coord_diff_no, y_coord_diff_no, z_coord_diff_no = distance_periodicity(n, o)
    euclidean_distance_hnn, x_coord_diff_hnn, y_coord_diff_hnn, z_coord_diff_hnn = distance_periodicity(hn, n)
    euclidean_distance_hno, x_coord_diff_hno, y_coord_diff_hno, z_coord_diff_hno = distance_periodicity(hn, o)
    
    vector_hnn = [x_coord_diff_hnn, y_coord_diff_hnn, z_coord_diff_hnn]
    vector_hno = [x_coord_diff_hno, y_coord_diff_hno, z_coord_diff_hno]

    angle = calc_angle(vector_hno, vector_hnn)

    # print(angle)

    if (angle > 2.35619) and  (angle <= 3.14159) and euclidean_distance_no <= 3.5:
        return True
    return False

def getHBinformation(timeframeCoordinates, cluster, timeframeNumber): #return the HB information for a single cluster
    atomArrays = getAtomTypesFromArray(cluster.indices)

    for hn in atomArrays["hn"]:
        for n in atomArrays["n"]:
            coord_hn = getCoordinates(timeframeCoordinates, hn)
            coord_n = getCoordinates(timeframeCoordinates, n)
            for o in atomArrays["o"]:
                coord_o = getCoordinates(timeframeCoordinates, o)
                if HB_truth(coord_hn, coord_n, coord_o):
                    cluster.hb_list.append(HydrogenBond(hn, coord_hn, n, coord_n, o, coord_o, timeframeNumber, "o"))
                    cluster.hb_present = True
                    cluster.totalHB += 1
            for os in atomArrays["os"]:
                coord_os = getCoordinates(timeframeCoordinates, os)
                if HB_truth(coord_hn, coord_n, coord_os):
                    cluster.hb_list.append(HydrogenBond(hn, coord_hn, n, coord_n, os, coord_os, timeframeNumber, "os"))
                    cluster.hb_present = True
                    cluster.totalHB += 1
    return cluster


def getHBinformationGivenIndices(timeframeCoordinates, indices, timeframeNumber): #return the HB information for a single cluster
    atomArrays = getAtomTypesFromArray(indices)
    HBs = []
    presence = False

    for hn in atomArrays["hn"]:
        for n in atomArrays["n"]:
            coord_hn = getCoordinates(timeframeCoordinates, hn)
            coord_n = getCoordinates(timeframeCoordinates, n)
            for o in atomArrays["o"]:
                coord_o = getCoordinates(timeframeCoordinates, o)
                if HB_truth(coord_hn, coord_n, coord_o):
                    HBs.append(HydrogenBond(hn, coord_hn, n, coord_n, o, coord_o, timeframeNumber, "o"))
                    presence = True
            for os in atomArrays["os"]:
                coord_os = getCoordinates(timeframeCoordinates, os)
                if HB_truth(coord_hn, coord_n, coord_os):
                    HBs.append(HydrogenBond(hn, coord_hn, n, coord_n, os, coord_os, timeframeNumber, "os"))
                    presence = True
    return presence, HBs


